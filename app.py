import streamlit as st
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# ---------------- UI ----------------
st.set_page_config(page_title="Market Intelligence Assistant")
st.title("📊 Market Intelligence Assistant")
st.write("💡 Ask questions about market trends, industries, or business insights")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("market_intelligence_dataset.csv")

df = load_data()

# ---------------- LOAD EMBEDDING MODEL ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# Precompute embeddings
@st.cache_resource
def compute_embeddings(texts):
    return embedder.encode(texts, convert_to_tensor=True)

texts = df["text"].tolist()
embeddings = compute_embeddings(texts)

# ---------------- LOAD LLM ----------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- RAG FUNCTION ----------------
def rag_chain(question):
    query_embedding = embedder.encode(question, convert_to_tensor=True)

    # Similarity search
    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(scores, k=3)

    context = "\n".join([texts[idx] for idx in top_results.indices])

    prompt = f"""
You are a financial analyst.

Answer clearly based on the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in response:
        response = response.split("Answer:")[-1]

    return response.strip()

# ---------------- INPUT ----------------
query = st.text_input("🔍 Enter your question:")

# ---------------- OUTPUT ----------------
if query:
    with st.spinner("Analyzing market data..."):
        answer = rag_chain(query)

    st.subheader("📌 Answer")
    st.success(answer)
