import json
import random
from datasets import load_dataset

# Configuration
TOTAL_TRAIN = 8000
TOTAL_VAL = 2000
OUTPUT_TRAIN = "generated_dataset.jsonl"
OUTPUT_VAL = "generated_dataset_validation.jsonl"

def process_cnn(item):
    article = item['article']
    # Filter out empty or super long articles
    if not article or len(article) > 5000: return None
    return {
        "instruction": f"Summarize this news article: {article}",
        "response": item['highlights']
    }

print("Downloading CNN/DailyMail dataset...")
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Prepare Training Data
print("Preparing Training Data...")
train_data = []
# We shuffle to get random articles
shuffled_train = dataset['train'].shuffle(seed=42)

for item in shuffled_train:
    if len(train_data) >= TOTAL_TRAIN: break
    entry = process_cnn(item)
    if entry: train_data.append(entry)

with open(OUTPUT_TRAIN, "w") as f:
    for entry in train_data:
        f.write(json.dumps(entry) + "\n")

# Prepare Validation Data
print("Preparing Validation Data...")
val_data = []
shuffled_val = dataset['validation'].shuffle(seed=42)

for item in shuffled_val:
    if len(val_data) >= TOTAL_VAL: break
    entry = process_cnn(item)
    if entry: val_data.append(entry)

with open(OUTPUT_VAL, "w") as f:
    for entry in val_data:
        f.write(json.dumps(entry) + "\n")

print(f"SUCCESS! Created {len(train_data)} training samples and {len(val_data)} validation samples.")