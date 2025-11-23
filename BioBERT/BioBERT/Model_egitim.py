from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
import torch


print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


df = pd.read_csv("biobert_ddi_dataset.csv")

label_list = sorted(df["label"].unique())
label_to_id = {label: i for i, label in enumerate(label_list)}
df["label_id"] = df["label"].map(label_to_id)

df = df.drop(columns=["label"])
df = df.rename(columns={"label_id": "labels"})

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.15, seed=42)


tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

def encode(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=192  
    )

dataset = dataset.map(encode, batched=True)


model = AutoModelForSequenceClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
    num_labels=len(label_list),
    use_safetensors=True
)


training_args = TrainingArguments(
    output_dir="./biobert_ddi_model",
    per_device_train_batch_size=2,         
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,          
    fp16=True,                             
    num_train_epochs=3,
    learning_rate=2e-5,
    warmup_steps=200,
    logging_steps=100,
    save_steps=2000,
    eval_steps=2000,
    load_best_model_at_end=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()

trainer.save_model("./biobert_ddi_model")
print("Model başarıyla kaydedildi.")
