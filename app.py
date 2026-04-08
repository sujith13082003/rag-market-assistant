import streamlit as st
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

# ---------------- UI ----------------
st.set_page_config(page_title="Market Intelligence Assistant", layout="centered")

st.title("📊 Market Intelligence Assistant")
st.write("💡 Ask questions about market trends, industries, or business insights")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("market_intelligence_dataset.csv")

df = load_data()

# ---------------- CREATE DOCUMENTS ----------------
@st.cache_resource
def create_vector_store(df):
    documents = []
    
    for _, row in df.iterrows():
        content = f"Sentiment: {row['sentiment']}\nNews: {row['text']}"
        documents.append(Document(page_content=content))

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    return db

db = create_vector_store(df)
retriever = db.as_retriever()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=128
    )

    return HuggingFacePipeline(pipeline=pipe)

llm = load_model()

# ---------------- PROMPT ----------------
prompt = PromptTemplate.from_template(
    """You are a financial analyst.

Answer clearly and concisely based only on the context below.

Context:
{context}

Question:
{question}

Answer:"""
)

# ---------------- RAG FUNCTION ----------------
def rag_chain(question):
    docs = retriever.invoke(question)

    context = "\n".join([doc.page_content for doc in docs])

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": context,
        "question": question
    })

    # Clean output
    if "Answer:" in response:
        response = response.split("Answer:")[-1]

    return response.strip()

# ---------------- INPUT UI ----------------
query = st.text_input("🔍 Enter your question:")

# ---------------- OUTPUT ----------------
if query:
    with st.spinner("Analyzing market data..."):
        answer = rag_chain(query)

    st.subheader("📌 Answer")
    st.success(answer)
