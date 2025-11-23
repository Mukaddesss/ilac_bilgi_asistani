import pandas as pd

clean = pd.read_csv("biobert_ddi_clean.csv")
dataset = pd.read_csv("biobert_ddi_dataset.csv")

# 1) İki dataset'i birleştir
merged = pd.concat([clean, dataset], ignore_index=True)

# 2) Duplicate temizliği
merged.drop_duplicates(subset=["text", "label"], inplace=True)

# 3) Gereksiz boşluk/format temizliği
merged["text"] = merged["text"].astype(str).str.strip()
merged["label"] = merged["label"].astype(str).str.strip()

# 4) Sınıf dağılımını hesapla
print("Class distribution:")
print(merged["label"].value_counts())

# 5) Final CSV kaydet
merged.to_csv("ddi_merged_clean.csv", index=False)
print("Merged dataset saved: ddi_merged_clean.csv")
