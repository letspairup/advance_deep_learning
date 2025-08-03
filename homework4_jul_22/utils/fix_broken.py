# fix_broken_qa.py
import os
import json

qa_path = "../data/train/balanced_qa_pairs.json"
output_path = "homework4_jul_22/data/train/balanced_qa_pairs_cleaned.json"

with open(qa_path, "r") as f:
    data = json.load(f)

filtered = []
missing = []

for item in data:
    if os.path.exists(item["image_path"]):
        filtered.append(item)
    else:
        missing.append(item["image_path"])

print(f"✅ Found {len(filtered)} valid QA pairs")
print(f"⚠️ Removed {len(missing)} entries due to missing image files")

with open(output_path, "w") as f:
    json.dump(filtered, f, indent=2)
