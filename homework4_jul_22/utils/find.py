import json
from pathlib import Path

data_dir = Path("../data/train")

valid_files = []

for file in data_dir.glob("*_info.json"):
    with open(file) as f:
        try:
            data = json.load(f)
            if "cameras" in data:
                valid_files.append(file.name)
        except:
            continue

print("✅ Valid files with 'cameras':")
for f in valid_files:
    print(f)
