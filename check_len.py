
import os
import json

dataset_dir = "Base_L"

for filename in os.listdir(dataset_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(dataset_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                print(f"{filename}: {len(data)} entries")
            except json.JSONDecodeError as e:
                print(f"Error decoding {filename}: {e}")
