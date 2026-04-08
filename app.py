import streamlit as st
import pandas as pd

st.set_page_config(page_title="Market Intelligence Assistant")

st.title("📊 Market Intelligence Assistant")
st.write("💡 Ask questions about market trends, industries, or business insights")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("market_intelligence_dataset.csv")

df = load_data()

# Simple search function
def get_answer(query):
    query = query.lower()

    results = df[df["text"].str.lower().str.contains(query)]

    if results.empty:
        return "No relevant market insights found. Try a different query."

    top_results = results.head(3)

    combined = " ".join(top_results["text"].tolist())

    return combined

# Input
query = st.text_input("🔍 Enter your question:")

# Output
if query:
    with st.spinner("Analyzing..."):
        answer = get_answer(query)

    st.subheader("📌 Answer")
    st.success(answer)
