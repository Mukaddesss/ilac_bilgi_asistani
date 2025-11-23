import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DATA_PATH = "ddi_merged_clean.csv"
MODEL_PATH = "./biobert_ddi_model_focal"   
OUTPUT_DIR = "./biobert_ddi_finetuned"
MAX_LEN = 192
BATCH_SIZE = 8
EPOCHS = 2
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

label_map = {
    "BENEFICIAL": 0,
    "LIFE-THREATENING": 1,
    "MINOR": 2,
    "MODERATE": 3,
    "SERIOUS": 4
}
inv_label_map = {v: k for k, v in label_map.items()}

class DDIDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }



print("\n📌 Loading dataset...")
df = pd.read_csv(DATA_PATH)
df["label_id"] = df["label"].map(label_map)

train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label_id"], random_state=42)


print("\n📌 Loading local BioBERT model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=5
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(DDIDataset(train_df, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(DDIDataset(val_df, tokenizer), batch_size=BATCH_SIZE, shuffle=False)

print("\n🚀 Starting training...\n")

for epoch in range(EPOCHS):
    model.train()
    train_losses = []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=mask)
        logits = outputs.logits

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    print(f"\n🔹 Epoch {epoch+1} Loss: {np.mean(train_losses):.4f}")

    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=mask).logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()

            preds.extend(pred)
            trues.extend(labels.cpu().numpy())

    acc = accuracy_score(trues, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average="macro")
    cm = confusion_matrix(trues, preds)

    print(f"🔹 Accuracy: {acc:.4f}")
    print(f"🔹 Precision: {prec:.4f}")
    print(f"🔹 Recall: {rec:.4f}")
    print(f"🔹 F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\n" + "=" * 60 + "\n")

print("\n💾 Saving fine-tuned model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✔ Saved to", OUTPUT_DIR)
