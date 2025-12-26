import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)

# Configuration
model_name = "distilgpt2"
max_len = 512
epochs = 3
batch_size = 8  # T4 GPU can handle this easily

class SummarizationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tok = tokenizer
        self.max_length = max_length
        self.data = []
        with open(file_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inst = self.data[idx]["instruction"].strip()
        resp = self.data[idx]["response"].strip()
        full_text = f"{inst}\nResponse: {resp}{self.tok.eos_token}"

        encodings = self.tok(
            full_text, truncation=True, max_length=self.max_length, padding="max_length"
        )
        input_ids = torch.tensor(encodings["input_ids"])
        return {"input_ids": input_ids, "labels": input_ids.clone()}

# Load Model & Tokenizer
print("Loading Model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load Datasets (created in Cell 3)
train_ds = SummarizationDataset("generated_dataset.jsonl", tokenizer, max_len)
val_ds = SummarizationDataset("generated_dataset_validation.jsonl", tokenizer, max_len)

# Training Setup
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,  # Uses GPU acceleration!
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=1,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print("Starting Training...")
trainer.train()

# Save the final model to Google Drive
final_path = "/content/drive/My Drive/ARAMI_Project/final_model"
print(f"Saving model to Google Drive at: {final_path}")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
print("DONE! Model saved safely.")
