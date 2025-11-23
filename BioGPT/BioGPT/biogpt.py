#pip install -q transformers torch chromadb sentence-transformers pandas accelerate sacremoses

from transformers import BioGptTokenizer, BioGptForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
import torch

print("BioGPT yükleniyor...")
model_name = "microsoft/biogpt"
tokenizer = BioGptTokenizer.from_pretrained(model_name)
model = BioGptForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print(f"BioGPT hazır! Cihaz: {device}")

print("Embedding modeli yükleniyor...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./biogpt_db_full")
try:
    client.delete_collection(name="ilaclar")
    print("Eski koleksiyon silindi")
except:
    pass
collection = client.create_collection(name="ilaclar")

print("CSV dosyası yükleniyor...")

csv_path = "ilaç_yeni.csv"

try:
    df = pd.read_csv(csv_path)
    print(f"{len(df)} ilaç yüklendi!")
    print(f"\nSütunlar: {df.columns.tolist()}")
    print(f"\nİlk 2 satır:\n{df.head(2)}")
except FileNotFoundError:
    print(f"HATA: '{csv_path}' dosyası bulunamadı!")
    print("\nLütfen dosyayı yükleyin:")
    print("   Google Colab'da: Sol menüden 'Files' -> dosyayı sürükle-bırak")
    print("   Sonra: csv_path = '/content/dosya_adi.csv'")
    raise

print("\nVeri temizleniyor...")

df['product_name'] = df['product_name'].fillna('Unknown')
df['salt_composition'] = df['salt_composition'].fillna('N/A')
df['sub_category'] = df['sub_category'].fillna('General')
df['medicine_desc'] = df['medicine_desc'].fillna('No description available')
df['side_effects'] = df['side_effects'].fillna('No side effects listed')
df['drug_interactions'] = df['drug_interactions'].fillna('{}')

print(f"Veri temizlendi. Toplam {len(df)} ilaç işlenecek.")

print("\nİlaçlar ChromaDB'ye ekleniyor...")
print("Bu işlem 7400 ilaç için 5-10 dakika sürebilir...\n")

batch_size = 100
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]

    for idx, row in batch.iterrows():
        combined_text = f"""Drug Name: {row['product_name']}
Active Ingredients: {row['salt_composition']}
Drug Category: {row['sub_category']}
Medical Description: {row['medicine_desc']}
Common Side Effects: {row['side_effects']}
Drug Interactions: {row['drug_interactions']}"""

        embedding = embedding_model.encode(combined_text).tolist()

        collection.add(
            embeddings=[embedding],
            documents=[combined_text],
            metadatas=[{
                "name": row['product_name'],
                "side_effects": str(row['side_effects']),
                "description": str(row['medicine_desc']),
                "interactions": str(row['drug_interactions'])
            }],
            ids=[f"drug_{idx}"]
        )

    print(f"{min(i+batch_size, len(df))}/{len(df)} ilaç eklendi ({(min(i+batch_size, len(df))/len(df)*100):.1f}%)")

print(f"\nTüm {len(df)} ilaç başarıyla ChromaDB'ye eklendi!")

def generate_with_biogpt(prompt, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def ilac_asistan(kullanici_sorusu):
    soru_lower = kullanici_sorusu.lower()

    etkilesim_keywords = ['etkileşim', 'interaction', 've', 'and', 'ile', 'with', 'beraber', 'together', 'birlikte']

    if any(word in soru_lower for word in etkilesim_keywords):
        import re

        ayirac_pattern = r'\b(ve|and|ile|with|,)\b'
        parcalar = re.split(ayirac_pattern, kullanici_sorusu, flags=re.IGNORECASE)

        ilac_adlari = []
        for parca in parcalar:
            parca_temiz = parca.strip().lower()
            if parca_temiz not in ['ve', 'and', 'ile', 'with', ',', 'etkileşim', 'interaction', 'nedir', 'var', 'mı', 'mi', 'nasıl', 'what', 'is', 'are', 'the', 'of', 'a', 'an', '?', 'kullanılır', 'birlikte', 'beraber', 'together']:
                if len(parca_temiz) > 3:
                    ilac_adlari.append(parca.strip())

        if len(ilac_adlari) >= 2:
            ilac1 = ilac_adlari[0]
            ilac2 = ilac_adlari[1]

            print(f"Tespit edilen ilaçlar: {ilac1} ve {ilac2}")

            query_emb1 = embedding_model.encode(ilac1).tolist()
            results1 = collection.query(query_embeddings=[query_emb1], n_results=1)

            query_emb2 = embedding_model.encode(ilac2).tolist()
            results2 = collection.query(query_embeddings=[query_emb2], n_results=1)

            ilac1_meta = results1['metadatas'][0][0]
            ilac2_meta = results2['metadatas'][0][0]

            ilac1_tam_ad = ilac1_meta['name']
            ilac2_tam_ad = ilac2_meta['name']

            prompt = f"""Drug 1: {ilac1_tam_ad}
Drug 1 Interactions: {ilac1_meta['interactions']}

Drug 2: {ilac2_tam_ad}
Drug 2 Interactions: {ilac2_meta['interactions']}

Question: Can {ilac1_tam_ad} and {ilac2_tam_ad} be taken together? What is the interaction severity level (Serious, Moderate, Minor, or No Interaction)?
Answer: The interaction between these drugs is"""

            yanit = generate_with_biogpt(prompt, max_new_tokens=120)

            if "Answer:" in yanit:
                yanit = yanit.split("Answer:")[-1].strip()

            yanit_lower = yanit.lower()
            if 'serious' in yanit_lower or 'severe' in yanit_lower or 'major' in yanit_lower:
                seviye = "CİDDİ"
                uyari = "Bu iki ilaç birlikte kullanılmamalıdır! Acilen doktorunuza danışın!"
            elif 'moderate' in yanit_lower:
                seviye = "ORTA"
                uyari = "Bu iki ilacın birlikte kullanımı dikkat gerektirir. Mutlaka doktorunuza danışın."
            elif 'minor' in yanit_lower or 'mild' in yanit_lower:
                seviye = "HAFIF"
                uyari = "Hafif etkileşim olabilir. Doktorunuza bilgi verin."
            elif 'no interaction' in yanit_lower or 'not interact' in yanit_lower:
                seviye = "ETKİLEŞİM YOK"
                uyari = "Bilinen ciddi etkileşim yok, ancak yine de doktorunuza danışmanız önerilir."
            else:
                seviye = "BELİRSİZ"
                uyari = "Etkileşim seviyesi belirsiz. Mutlaka doktorunuza danışın!"

            return f"""İKİ İLAÇ ETKİLEŞİM ANALİZİ

İlaç 1: {ilac1_tam_ad}
İlaç 2: {ilac2_tam_ad}

ETKİLEŞİM SEVİYESİ: {seviye}

BioGPT Analizi:
{yanit}

UYARI: {uyari}

ÖNEMLİ NOTLAR:
Bu bilgiler genel referans içindir
Her hastanın durumu farklıdır
Dozaj ve kullanım süresi etkileşimi değiştirebilir
Yeni ilaç başlatmadan MUTLAKA doktorunuza danışın"""

        else:
            query_embedding = embedding_model.encode(kullanici_sorusu).tolist()
            results = collection.query(query_embeddings=[query_embedding], n_results=1)

            ilac_bilgi = results['documents'][0][0]
            metadata = results['metadatas'][0][0]

            prompt = f"""{ilac_bilgi}

Question: What medications does this drug interact with?
Answer: This medication interacts with"""

            yanit = generate_with_biogpt(prompt, max_new_tokens=80)

            if "Answer:" in yanit:
                yanit = yanit.split("Answer:")[-1].strip()

            return f"""{metadata['name']} - Genel Etkileşim Bilgisi

BioGPT Özeti:
{yanit}

Veri Kaynağı:
{metadata['interactions']}

İKİ İLAÇ KARŞILAŞTIRMASI İÇİN:
"Parol ve Aspirin etkileşimi nedir?" gibi sorun.

Unutmayın: Kullandığınız tüm ilaçları doktorunuza bildirin."""

    query_embedding = embedding_model.encode(kullanici_sorusu).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1)

    ilac_bilgi = results['documents'][0][0]
    metadata = results['metadatas'][0][0]

    if any(word in soru_lower for word in ['side effect', 'yan etki', 'adverse', 'effect']):
        prompt = f"""{ilac_bilgi}

Question: What are the side effects of this medication?
Answer: The common side effects include"""

        yanit = generate_with_biogpt(prompt, max_new_tokens=60)

        if "Answer:" in yanit:
            yanit = yanit.split("Answer:")[-1].strip()

        return f"""{metadata['name']} - Yan Etkiler

BioGPT Analizi:
{yanit}

Veri Kaynağı: {metadata['side_effects']}

Önemli: Yan etkiler kişiden kişiye değişir. Ciddi yan etkiler görürseniz hemen doktorunuza başvurun."""

    elif any(word in soru_lower for word in ['usage', 'how to use', 'nasıl kullan', 'kullanım', 'how to take']):
        prompt = f"""{ilac_bilgi}

Question: How should this medication be used?
Answer: This medication should be"""

        yanit = generate_with_biogpt(prompt, max_new_tokens=80)

        if "Answer:" in yanit:
            yanit = yanit.split("Answer:")[-1].strip()

        return f"""{metadata['name']} - Kullanım Talimatları

BioGPT Önerisi:
{yanit}

Detaylı Açıklama: {metadata['description']}

Doktorunuzun önerdiği dozajı takip edin."""

    else:
        prompt = f"""{ilac_bilgi}

Provide a brief medical summary of this drug:"""

        yanit = generate_with_biogpt(prompt, max_new_tokens=100)

        return f"""{metadata['name']} - Genel Bilgi

BioGPT Özeti:
{yanit}

Daha spesifik sorular sorabilirsiniz:
"{metadata['name']} yan etkileri nelerdir?"
"{metadata['name']} nasıl kullanılır?"
"Parol ve Aspirin etkileşimi nedir?"

Bu bilgiler eğitim amaçlıdır. Medikal kararlar için doktorunuza danışın."""

print("\nSİSTEM HAZIR! Etkileşim seviyesi analizi aktif.")

print("="*70)
print(" İNTERAKTİF MOD")
print("="*70)
print("\n Örnek sorular:")
print("- 'Aspirin yan etkileri nelerdir?'")
print("- 'Paracetamol nasıl kullanılır?'")
print("- 'Metformin drug interactions?'")
print("\nÇıkmak için 'quit' veya 'exit' yazın.\n")

while True:
    try:
        soru = input(" Sorunuz: ").strip()

        if soru.lower() in ['quit', 'exit', 'çık', 'q']:
            print(" Görüşmek üzere!")
            break

        if not soru:
            print(" Lütfen bir soru yazın.\n")
            continue

        print(f"\n YANIT:")
        print(ilac_asistan(soru))
        print("\n" + "-"*70 + "\n")

    except KeyboardInterrupt:
        print("\n Görüşmek üzere!")
        break