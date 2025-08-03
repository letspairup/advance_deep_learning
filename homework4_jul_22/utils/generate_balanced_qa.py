import os
import json
from tqdm import tqdm
from homework4_jul_22.homework.generate_qa import generate_qa_pairs

INPUT_DIR = "../data/train"
OUTPUT_QA_FILE = os.path.join(INPUT_DIR, "balanced_qa_pairs.json")

qa_pairs = []
for filename in tqdm(os.listdir(INPUT_DIR)):
    if filename.endswith("_info.json"):
        info_path = os.path.join(INPUT_DIR, filename)
        try:
            pairs = generate_qa_pairs(info_path, view_index=6)
            for q in pairs:
                if os.path.exists(q["image_path"]):  # double-check image exists
                    qa_pairs.append(q)
                else:
                    print(f"⚠️ Skipping (missing image): {q['image_path']}")
        except Exception as e:
            print(f"⚠️ Skipping {filename}: {e}")

print(f"✅ Generated {len(qa_pairs)} QA pairs")

# Save as balanced QA file
with open(OUTPUT_QA_FILE, "w") as f:
    json.dump(qa_pairs, f, indent=2)

print(f"✅ Saved to: {OUTPUT_QA_FILE}")
