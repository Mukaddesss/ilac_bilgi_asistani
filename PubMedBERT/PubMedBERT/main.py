import pandas as pd
import torch
import faiss
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

df = pd.read_csv("ilaç_yeni.csv")

def clean_field(text):
    if not text or str(text).lower() in ["", "none"]:
        return ""
  
    items = re.split(r',|;|\n', str(text))  
    items = [i.strip() for i in items if i.strip()]
    return ", ".join(items)


df['side_effects'] = df['side_effects'].apply(clean_field)
df['drug_interactions'] = df['drug_interactions'].apply(clean_field)
df['medicine_desc'] = df['medicine_desc'].apply(lambda x: x.strip() if isinstance(x, str) else "")

corpus_texts = []
for _, row in df.iterrows():
    text = f"""
sub_category: {row['sub_category']}
product_name: {row['product_name']}
composition: {row['salt_composition']}
manufactured: {row['product_manufactured']}
description: {row['medicine_desc']}
side_effects: {row['side_effects']}
interactions: {row['drug_interactions']}
"""
    corpus_texts.append(text)

print("Toplam belge:", len(corpus_texts))

embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

embeddings = embedding_model.encode(
    corpus_texts, convert_to_numpy=True, normalize_embeddings=True
)


dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

lm_name = "microsoft/phi-3-mini-4k-instruct"

lm_model = AutoModelForCausalLM.from_pretrained(
    lm_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(lm_name)

gen = pipeline(
    "text-generation",
    model=lm_model,
    tokenizer=tokenizer,
    max_new_tokens=200
)

field_map = {
    "side effect": "side_effects",
    "side effects": "side_effects",
    "interaction": "interactions",
    "interactions": "interactions",
    "composition": "composition",
    "description": "description",
    "usage": "description",
}

from rapidfuzz import process
import re

def format_answer(field, raw_text):
    if not raw_text:
        return "No evidence found in the dataset."

    if field == "side_effects":
        effects = re.split(r',|;', raw_text)
        effects = [e.strip() for e in effects if e.strip()]
        if len(effects) == 1:
            return f"The only reported side effect is {effects[0]}."
        elif len(effects) > 1:
            return f"Reported side effects include {', '.join(effects[:-1])}, and {effects[-1]}."
        else:
            return "No evidence found in the dataset."

    elif field == "interactions":
        interactions = re.split(r',|;', raw_text)
        interactions = [i.strip() for i in interactions if i.strip()]
        return f"Interactions: {', '.join(interactions)}." if interactions else "No interactions found."

    elif field == "description":
        return f"Usage/Description: {raw_text}"

    elif field == "composition":
        return f"Composition: {raw_text}"

    return raw_text


def detect_product_name_fuzzy(query, product_list, threshold=75):
    result = process.extractOne(query, product_list)
    if result is None:
        return None
    match = result[0]
    score = result[1]
    if score >= threshold:
        return match
    return None


def rag_medical_answer(query, k=5, return_contexts=False):

    product_name = detect_product_name_fuzzy(query, df['product_name'].unique().tolist(), threshold=75)
    if not product_name:
        return "Could not automatically detect the product name. Please include it in the question."

    query_lower = query.lower()


    if any(x in query_lower for x in ["how to use", "how should i use", "usage"]):
        raw_text = df[df['product_name'].str.lower() == product_name.lower()]['medicine_desc'].values
        answer = raw_text[0] if len(raw_text) > 0 else "No evidence found in the dataset."
        return f"Usage/Description: {answer}"


    field = None
    for key in field_map:
        if key in query_lower:
            field = field_map[key]
            break
    if not field:
        return "No evidence found in the dataset."


    q_emb = embedding_model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb, k)
    retrieved_texts = [corpus_texts[i] for i in I[0]]
    context = "\n\n".join(retrieved_texts)

    prompt = f"""
You are a strict medical assistant.
Answer ONLY using the information in the context below.
Do NOT add, infer, or guess information.
If no evidence exists, say exactly: "No evidence found in the dataset."

Context:
{context}

Question: {query}

###ANSWER:
"""
    raw_output = gen(prompt)[0]["generated_text"]


    match_field = re.search(fr"{field}:\s*([^\n]+)", context, re.IGNORECASE)
    answer = match_field.group(1).strip() if match_field else "No evidence found in the dataset."
    formatted = format_answer(field, answer)

    return (formatted, retrieved_texts) if return_contexts else formatted

print(rag_medical_answer("How should I use Ricetral 0.3gm Injection?"))
print(rag_medical_answer("What are the interactions of Bexol DT Tablet?"))
print(rag_medical_answer("What are the side effects of Bexol DT Tablet?"))

import re
from sklearn.metrics import precision_score, recall_score, f1_score


test_set = []
for _, row in df.iterrows():
    product_name = row['product_name']
    if row['medicine_desc'].strip():
        test_set.append((f"How should I use {product_name}?", "description", row['medicine_desc']))
    if row['side_effects'].strip():
        test_set.append((f"What are the side effects of {product_name}?", "side_effects", row['side_effects']))
    if row['drug_interactions'].strip():
        test_set.append((f"What are the interactions of {product_name}?", "interactions", row['drug_interactions']))


preds = []
golds = []
fields = []


for question, field, gold_text in test_set[:20]:

    product_name = detect_product_name_fuzzy(question, df['product_name'].unique().tolist(), threshold=75)
    if not product_name:
        pred = "No evidence found in the dataset."
    else:
        q_emb = embedding_model.encode([question], normalize_embeddings=True)
        D, I = index.search(q_emb, 5)
        context = "\n\n".join([corpus_texts[i] for i in I[0]])
        match_field = re.search(fr"{field}:\s*([^\n]+)", context, re.IGNORECASE)
        answer = match_field.group(1).strip() if match_field else "No evidence found in the dataset."
        pred = format_answer(field, answer)

    preds.append(pred)
    golds.append(gold_text)
    fields.append(field)


def token_set_f1(pred_text, gold_text):
    pred_tokens = set(re.findall(r'\w+', str(pred_text).lower()))
    gold_tokens = set(re.findall(r'\w+', str(gold_text).lower()))

    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0  # boşsa tam skor
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    tp = len(pred_tokens & gold_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


field_metrics = {}
for field in set(fields):
    precisions, recalls, f1s = [], [], []
    for i in range(len(preds)):
        if fields[i] == field:
            p, r, f = token_set_f1(preds[i], golds[i])
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
    field_metrics[field] = {
        "Precision": sum(precisions)/len(precisions),
        "Recall": sum(recalls)/len(recalls),
        "F1": sum(f1s)/len(f1s)
    }


for field, metrics in field_metrics.items():
    print(f"{field.upper()} -> Precision: {metrics['Precision']:.4f}, "
          f"Recall: {metrics['Recall']:.4f}, F1: {metrics['F1']:.4f}")

