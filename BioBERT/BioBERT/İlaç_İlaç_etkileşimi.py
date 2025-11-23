import re
import gradio as gr
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DATA_PATH = "ilaç_yeni.csv"
MODEL_PATH = "./biobert_ddi_finetuned"  

df = pd.read_csv(DATA_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

label_map = {0: "BENEFICIAL", 1: "LIFE-THREATENING", 2: "MINOR", 3: "MODERATE", 4: "SERIOUS"}


def detect_intent(q):
    q = q.lower()

    if "etkileşim" in q or ("ile" in q and "var mı" in q):
        return "interaction"
    if "yan etki" in q:
        return "side_effect"
    if "kullan" in q or "nasıl" in q or "ne işe yarar" in q:
        return "usage"

    return "usage"  #  fallback


def extract_drugs(question):
    found = []
    q = question.lower()

    for name in df["product_name"].dropna().tolist():
        n = str(name).lower()
        if n in q:
            found.append(name)

    return list(set(found))


# model etk. tahmini

def predict_interaction(drugA, drugB):
    descA = get_desc(drugA)
    descB = get_desc(drugB)

    text = f"""
    DrugA: {drugA}
    Desc: {descA}

    DrugB: {drugB}
    Desc: {descB}

    Analyze pharmacological interaction.
    """

    tokens = tokenizer(text, return_tensors="pt", truncation=True,
        padding=True, max_length=192)

    with torch.no_grad():
        out = model(**tokens)
        probs = F.softmax(out.logits, dim=1).numpy()[0]

    pred = int(probs.argmax())
    conf = float(probs[pred])

    return label_map[pred], conf

def get_desc(drug):
    try:
        return df[df["product_name"] == drug].iloc[0]["medicine_desc"]
    except:
        return "Bilgi bulunamadı."

def get_side(drug):
    try:
        return df[df["product_name"] == drug].iloc[0]["side_effects"]
    except:
        return "Bilgi bulunamadı."

def answer_user(question):
    drugs = extract_drugs(question)
    intent = detect_intent(question)

    if len(drugs) == 0:
        return "İlaç adı tespit edemedim. Lütfen daha net bir isim yaz."

   
    if len(drugs) == 1:
        d = drugs[0]

        if intent == "usage":
            return f"💊 **{d} Kullanım Bilgisi**\n\n{get_desc(d)}"

        if intent == "side_effect":
            return f"⚠️ **{d} Yan Etkileri**\n\n{get_side(d)}"

        return f"**{d} hakkında kullanım veya yan etki bilgisi isteniyor gibi algıladım ama tam anlayamadım.**"

    
    if len(drugs) >= 2:
        drugA, drugB = drugs[:2]
        label, conf = predict_interaction(drugA, drugB)

        return f"""
🔍 **{drugA} + {drugB} ETKİLEŞİM ANALİZİ**

**📌 Model Tahmini:** {label}  
**📌 Güven Skoru:** {conf:.3f}

---

### 💊 {drugA} Kullanım Bilgisi:
{get_desc(drugA)}

### 💊 {drugB} Kullanım Bilgisi:
{get_desc(drugB)}

---

### ⚠️ {drugA} Yan Etkileri:
{get_side(drugA)}

### ⚠️ {drugB} Yan Etkileri:
{get_side(drugB)}
"""


with gr.Blocks(title="İlaç Etkileşim & Kullanım Asistanı") as demo:

    gr.Markdown("""
    # 💊 İlaç Etkileşim & Kullanım Asistanı  
    Aşağıya bir soru yazın.  
    Örnekler:  
    - **Vifol nasıl kullanılır?**  
    - **Muzika yan etkileri neler?**  
    - **Vifol ile Muzika arasında etkileşim var mı?**  
    """)

    inp = gr.Textbox(label="Sorunuz")
    out = gr.Markdown(label="Cevap")

    btn = gr.Button("Gönder")
    btn.click(answer_user, inp, out)

demo.launch()
