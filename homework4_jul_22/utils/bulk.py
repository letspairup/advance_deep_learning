import os
import json
from pathlib import Path
from homework4_jul_22.homework.generate_qa import generate_qa_pairs

DATA_DIR = Path("../data/train")
count = 0

for info_file in DATA_DIR.glob("*_info.json"):
    with open(info_file) as f:
        try:
            info = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ Skipping {info_file.name}: invalid JSON")
            continue

        if "ego_name" not in info or "karts" not in info or "track" not in info:
            print(f"⚠️ Skipping {info_file.name}: missing required keys")
            continue

    try:
        base = info_file.stem.replace("_info", "")
        view_indices = range(len(info["detections"]))  # one QA file per view
        for view_index in view_indices:
            try:
                qa_pairs = generate_qa_pairs(str(info_file), view_index)
                if not qa_pairs:
                    continue

                output_path = DATA_DIR / f"{base}_{view_index:02d}_qa_pairs.json"
                with open(output_path, "w") as out:
                    json.dump(qa_pairs, out, indent=2)
                count += 1
            except Exception as e:
                print(f"❌ Failed for {info_file.name} view {view_index}: {e}")
    except Exception as e:
        print(f"❌ Skipped {info_file.name}: {e}")

print(f"✅ Generated {count} QA files in {DATA_DIR}")
