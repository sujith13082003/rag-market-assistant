import streamlit as st
import pandas as pd

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

st.title("📊 Market Intelligence Assistant")

st.write("💡 Ask questions about market trends, industries, or business insights")

# 🔥 LOAD DATASET
df = pd.read_csv("market_intelligence_dataset.csv")

# 🔥 CONVERT TO DOCUMENTS
documents = []
for _, row in df.iterrows():
    content = f"Sentiment: {row['sentiment']}\nNews: {row['text']}"
    documents.append(Document(page_content=content))

# 🔥 SPLIT TEXT
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 🔥 CREATE EMBEDDINGS + VECTOR DB
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever()

# 🔥 LOAD MODEL (FIXED TASK)
pipe = pipeline(
    "text2text-generation",   # ✅ IMPORTANT FIX
    model="google/flan-t5-small",
    max_length=128
)

llm = HuggingFacePipeline(pipeline=pipe)

# PROMPT
prompt = PromptTemplate.from_template(
    """You are a financial analyst.

Answer clearly based only on the context.

Context:
{context}

Question:
{question}

Answer:"""
)

# RAG FUNCTION
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

# INPUT
query = st.text_input("🔍 Enter your question here:")

# OUTPUT
if query:
    with st.spinner("Analyzing..."):
        answer = rag_chain(query)
    st.success(answer)
