import os
import json
import random
import fire
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

FALLBACK_EGO_NAME = "tux"

def get_all_info_files(data_dir):
    return [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if fname.endswith("_info.json")
    ]


def generate_question_track_name(info):
    return {
        "question": "What track is this?",
        "answer": info["track"],
    }


def generate_question_ego_kart_name(info, ego_kart):
    return {
        "question": "What kart is the ego car?",
        "answer": ego_kart,
    }


def generate_question_num_karts(info):
    return {
        "question": "How many karts are there in the scenario?",
        "answer": str(len(info["karts"])),
    }


def generate_question_num_karts_in_front(info, view_index, ego_kart):
    ego_idx = info["karts"].index(ego_kart)
    ego_pos = info["distance_down_track"][ego_idx]
    count = sum(1 for dist in info["distance_down_track"] if dist > ego_pos)
    return {
        "question": "How many karts are in front of the ego car?",
        "answer": str(count),
    }


def generate_question_num_karts_behind(info, view_index, ego_kart):
    ego_idx = info["karts"].index(ego_kart)
    ego_pos = info["distance_down_track"][ego_idx]
    count = sum(1 for dist in info["distance_down_track"] if dist < ego_pos)
    return {
        "question": "How many karts are behind the ego car?",
        "answer": str(count),
    }


def generate_question_is_front_or_back(info, view_index, ego_kart, kart):
    ego_idx = info["karts"].index(ego_kart)
    kart_idx = info["karts"].index(kart)
    ans = "front" if info["distance_down_track"][kart_idx] > info["distance_down_track"][ego_idx] else "back"
    return {
        "question": f"Is {kart} in front of or behind the ego car?",
        "answer": ans,
    }


def generate_qa_pairs(info, view_index):
    qa_pairs = []
    ego_kart = info.get("ego_name", FALLBACK_EGO_NAME)
    if "ego_name" not in info:
        print(f"⚠️  'ego_name' missing — using fallback '{FALLBACK_EGO_NAME}'")

    qa_pairs.append(generate_question_track_name(info))
    qa_pairs.append(generate_question_ego_kart_name(info, ego_kart))
    qa_pairs.append(generate_question_num_karts(info))

    if ego_kart in info["karts"]:
        qa_pairs.append(generate_question_num_karts_in_front(info, view_index, ego_kart))
        qa_pairs.append(generate_question_num_karts_behind(info, view_index, ego_kart))
        for kart in info["karts"]:
            if kart != ego_kart:
                qa_pairs.append(generate_question_is_front_or_back(info, view_index, ego_kart, kart))

    return qa_pairs


def check_qa_pairs(info_file, view_index=6):
    with open(info_file, "r") as f:
        info = json.load(f)
    image_file = info_file.replace("_info.json", f"_{view_index:02d}_im.jpg")
    image = Image.open(image_file)

    qa_pairs = generate_qa_pairs(info, view_index)
    print(f"✅ QA Pairs generated: {len(qa_pairs)}")
    for qa in qa_pairs:
        print(f"Q: {qa['question']} -> A: {qa['answer']}")

    plt.imshow(image)
    plt.title("View Index: {}".format(view_index))
    plt.axis("off")
    plt.show()


def run(info_file, view_index=6):
    with open(info_file, "r") as f:
        info = json.load(f)
    qa_pairs = generate_qa_pairs(info, view_index)
    print(json.dumps(qa_pairs, indent=2))


def bulk_generate(data_dir="homework4_jul_22/data/train", out_file="homework4_jul_22/data/train/balanced_qa_pairs.json"):
    all_info_files = get_all_info_files(data_dir)
    all_pairs = []

    for path in tqdm(all_info_files):
        try:
            with open(path, "r") as f:
                info = json.load(f)
            qa_pairs = generate_qa_pairs(info, view_index=6)
            image_path = path.replace("_info.json", "_06_im.jpg")
            for pair in qa_pairs:
                pair["image_path"] = image_path
                all_pairs.append(pair)
        except Exception as e:
            print(f"❌ Skipping {path}: {e}")
            continue

    with open(out_file, "w") as f:
        json.dump(all_pairs, f, indent=2)
    print(f"✅ Saved {len(all_pairs)} QA pairs to {out_file}")


def main():
    fire.Fire({
        "check": check_qa_pairs,
        "run": run,
        "bulk_generate": bulk_generate,
    })


if __name__ == "__main__":
    main()
