import pandas as pd
import json

# KENDİ DOSYA ADINI YAZ
df = pd.read_csv("ilaç_yeni.csv")

dataset = []

for idx, row in df.iterrows():
    drug1 = str(row["product_name"])
    active = str(row["salt_composition"])
    desc = str(row["medicine_desc"])
    
    interactions_raw = row["drug_interactions"]
    
    try:
        interactions = json.loads(interactions_raw)
    except:
        continue
    
    drugs = interactions.get("drug", [])
    effects = interactions.get("effect", [])
    
    for d, e in zip(drugs, effects):
        text = (
            f"Drug1: {drug1}\n"
            f"Active ingredient: {active}\n"
            f"Description: {desc}\n\n"
            f"Drug2: {d}"
        )
        
        dataset.append({
            "text": text,
            "label": e.upper().strip()
        })

dataset_df = pd.DataFrame(dataset)
dataset_df.to_csv("biobert_ddi_dataset.csv", index=False)

print("DATASET OLUŞTURULDU ↓↓↓")
print(dataset_df.head())
print("Toplam satır:", len(dataset_df))
