import os
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image

DATA_DIR = Path(__file__).parent.parent / "data"

def generate_captions(split: str = "valid", ckpt_path: str = "vlm_model"):

    from .finetune import load
    model = load(ckpt_path)

    image_dir = DATA_DIR / split
    image_paths = sorted(image_dir.glob("*.jpg"))

    print(f"Found {len(image_paths)} images in {split}")

    results = []
    for image_path in tqdm(image_paths, desc=f"Generating captions for {split}"):
        image = Image.open(image_path).convert("RGB")

        try:
            prompt = "Describe this image."
            output = model.answer([image], [prompt])
            caption = output[0].strip()
        except Exception as e:
            print(f"❌ Failed for {image_path.name}: {e}")
            caption = ""

        if caption:
            results.append({
                "image_file": f"{split}/{image_path.name}",
                "caption": caption
            })

    out_path = DATA_DIR / split / f"{split}_captions.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Captions saved to: {out_path}")


if __name__ == "__main__":
    from fire import Fire
    Fire(generate_captions)
