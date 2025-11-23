import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import pandas as pd
from datasets import Dataset


print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


class FocalLoss(nn.Module):
    """
    Dengesiz veri setleri için Focal Loss.
    gamma: zor örneklere ne kadar ağırlık verileceğini kontrol eder (genelde 2.0 iyi).
    weight: istersen class weight de verebilirsin (şimdilik None bırakıyoruz).
    """

    def __init__(self, gamma: float = 2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, labels):
        # logits: (batch_size, num_labels)
        # labels: (batch_size,)
        ce_loss = F.cross_entropy(
            logits, labels, weight=self.weight, reduction="none"
        )  # her örnek için CE loss
        pt = torch.exp(-ce_loss)  # p_t
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()



class CustomTrainer(Trainer):
    def __init__(self, focal_loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = focal_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")  # silme yok, pop kullanma
        # model'den logits al
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # focal loss uygula
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


df = pd.read_csv("biobert_ddi_dataset.csv")

print("\nLabel dağılımı:")
print(df["label"].value_counts())


label_list = sorted(df["label"].unique())
print("\nlabel_list (alfabetik):", label_list)

label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}
print("\nlabel_to_id:", label_to_id)
print("id_to_label:", id_to_label)


df["labels"] = df["label"].map(label_to_id)

df = df.drop(columns=["label"])

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.15, seed=42)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print("\nTrain örnek sayısı:", len(train_dataset))
print("Eval örnek sayısı:", len(eval_dataset))


MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def encode(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=192,  
    )


train_dataset = train_dataset.map(encode, batched=True)
eval_dataset = eval_dataset.map(encode, batched=True)

cols_to_keep = ["input_ids", "attention_mask", "labels"]
if "token_type_ids" in train_dataset.column_names:
    cols_to_keep.append("token_type_ids")

train_dataset.set_format(type="torch", columns=cols_to_keep)
eval_dataset.set_format(type="torch", columns=cols_to_keep)



num_labels = len(label_list)
print("\nnum_labels:", num_labels)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    use_safetensors=True, 
)

model.config.label2id = label_to_id
model.config.id2label = id_to_label


training_args = TrainingArguments(
    output_dir="./biobert_ddi_model_focal",
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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

focal_loss = FocalLoss(gamma=2.0) 

trainer = CustomTrainer(
    focal_loss=focal_loss,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)


trainer.train()


save_path = "./biobert_ddi_model_focal"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print("\nFocal Loss ile eğitilmiş model başarıyla kaydedildi:", save_path)
print("Sınıflar:", id_to_label)
