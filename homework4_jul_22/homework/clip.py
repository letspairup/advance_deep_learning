from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import CaptionDataset, MultiChoiceQADataset

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class CLIP(nn.Module):
    def __init__(self, vision_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int = 64, temperature: float = 0.07):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.vision_proj = nn.Linear(self.vision_encoder.config.hidden_size, proj_dim, bias=False)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, proj_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor([1.0 / temperature]))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image_outputs = self.vision_encoder(pixel_values=image)
        image_feat = image_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.vision_proj(image_feat)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.text_proj(text_feat)

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None, **kwargs):
        image_feat = self.encode_image(pixel_values)
        text_feat = self.encode_text(input_ids, attention_mask)
        image_feat = nn.functional.normalize(image_feat, dim=-1)
        text_feat = nn.functional.normalize(text_feat, dim=-1)
        logits_per_image = self.logit_scale.exp() * image_feat @ text_feat.t()
        logits_per_text = logits_per_image.t()
        return image_feat, text_feat, logits_per_image

    def save_pretrained(self, save_directory: str, **kwargs):
        additional_state_dict = {}
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            additional_state_dict[name] = param.data
        torch.save(additional_state_dict, Path(save_directory) / "additional_weights.pt")

    def load_pretrained(self, load_directory: str, **kwargs):
        additional_weights_path = Path(load_directory) / "additional_weights.pt"
        if additional_weights_path.exists():
            additional_state_dict = torch.load(additional_weights_path, map_location="cpu")
            for name, param in self.named_parameters():
                if "vision_encoder." in name or "text_encoder." in name:
                    continue
                param.data = additional_state_dict[name]

    def set_trainable_parameters(self):
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            param.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs):
        self.vision_encoder.gradient_checkpointing_enable(**kwargs)
        self.text_encoder.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)
        self.vision_encoder.embeddings.register_forward_hook(make_inputs_require_grads)
        self.text_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grads)


def compute_clip_loss(outputs, labels, num_items_in_batch=None):
    image_feat, text_feat, logits_per_image = outputs
    logits_per_text = logits_per_image.t()
    batch_size = logits_per_image.size(0)
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = nn.functional.cross_entropy(logits_per_image, labels)
    loss_t = nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


class CaptionDatasetForTraining(Dataset):
    def __init__(self, dataset: CaptionDataset, processor: AutoProcessor):
        self.dataset = dataset
        self.image_processor = tv.transforms.Compose([
            tv.transforms.Resize(192),
            tv.transforms.RandomResizedCrop(192, scale=(0.5, 1.0)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.image_processor(image)
        text = item["caption"] + self.processor.tokenizer.eos_token
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        input_ids = text_inputs["input_ids"].squeeze(0).long()
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }


def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_length = max(f["input_ids"].shape[0] for f in features)
    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])
    input_ids = torch.stack([pad_tensor(f["input_ids"], processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], 0) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = input_ids.clone()
    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "pixel_values": pixel_values.float(),
        "labels": labels.long(),
    }


def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("vision_encoder" in name or "text_encoder" in name) and "projection" not in name:
            target_modules.append(name)
    return target_modules


def train(
        data_dir: Path | None = None,
        output_dir: str = "clip",
        num_train_epochs: float = 1,
        per_device_train_batch_size: int = 1024,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-4,
        num_workers: int = 16,
):
    vlm = BaseVLM()
    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir / "tensorboard")
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    model = CLIP(vision_encoder, text_encoder).to(device).bfloat16()
    model.set_trainable_parameters()
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=get_target_modules_for_lora(model),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    train_dataset = CaptionDataset("train", data_dir)
    train_dataset = CaptionDatasetForTraining(train_dataset, processor)
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        bf16=True if device == "cuda" else False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        label_names=["labels"],
        dataloader_num_workers=num_workers,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
        compute_loss_func=compute_clip_loss,
    )
    trainer.train()
    trainer.save_model(output_dir)
    model.model.save_pretrained(output_dir)
    writer.close()
    return model, processor


def demo_train():
    train(
        train_dataset_name="train_demo",
        output_dir="demo_clip",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        num_workers=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-8,
    )


def test(ckpt_path: str, val_dataset: str = "valid_grader"):
    import tqdm
    testset = MultiChoiceQADataset(val_dataset)
    clip = load(ckpt_path).to(device)
    image_processor = tv.transforms.Compose([
        tv.transforms.Resize(192),
        tv.transforms.CenterCrop(192),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    correct_count = 0
    total_count = 0
    for pair in tqdm.tqdm(testset):
        image = Image.open(pair["image_path"]).convert("RGB")
        pixel_values = image_processor(image).unsqueeze(0).to(device).bfloat16()
        text_inputs = processor(
            text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = text_inputs["input_ids"].long().to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        vision_feature, text_feature, _ = clip(pixel_values, input_ids, attention_mask)
        prediction = torch.matmul(vision_feature, text_feature.T).argmax(dim=-1)
        if prediction == pair["correct_index"]:
            correct_count += 1
        total_count += 1
    print(f"Accuracy: {correct_count / total_count}")

def load(model_name: str = "clip") -> CLIP:
    from pathlib import Path
    model_path = Path(__file__).parent / model_name

    # Load vision and text encoder from BaseVLM
    vlm = BaseVLM()
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model

    # Init and load CLIP model
    clip_model = CLIP(vision_encoder, text_encoder).to(device)
    clip_model.load_pretrained(model_path)
    clip_model.eval()
    return clip_model

def main():
    from fire import Fire
    Fire({"train": train, "test": test})


if __name__ == "__main__":
    main()
