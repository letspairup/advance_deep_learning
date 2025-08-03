import json
import random
from pathlib import Path

OUTPUT_DIR = Path("../data/train")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRACKS = ["snowmountain", "zengarden", "cocoa_temple", "abyss", "volcano_island"]
KARTS = ["tux", "sara_the_wizard", "beastie", "amanda", "wilber", "adiumy", "hexley", "gnu"]

def generate_info(index):
    num_karts = random.randint(3, 6)
    selected_karts = random.sample(KARTS, num_karts)
    ego = selected_karts[0]

    info = {
        "ego_name": ego,
        "track": random.choice(TRACKS),
        "karts": [{"name": k, "relative_position": random.choice(["left", "right", "front", "back"])} for k in selected_karts],
        "cameras": [{}],  # dummy view
        "detections": [
            [[1, i, random.randint(100,300), random.randint(100,200), random.randint(300,400), random.randint(200,300)]
             for i in range(num_karts)]
        ]
    }

    filename = OUTPUT_DIR / f"synthetic_{index:03d}_info.json"
    with open(filename, "w") as f:
        json.dump(info, f, indent=2)

    return filename

# Generate 50 synthetic info files
generated_files = []
for i in range(500):
    file = generate_info(i)
    generated_files.append(file)

print(f"✅ Generated {len(generated_files)} synthetic *_info.json files in {OUTPUT_DIR}")
