import streamlit as st
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("📊 Market Intelligence Assistant")

st.write("💡 Ask questions about market trends, industries, or business insights")

# Load DB
embeddings = HuggingFaceEmbeddings()
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever()

# Load model
pipe = pipeline(
    "text-generation",
    model="google/flan-t5-small",
    max_length=256
)

llm = HuggingFacePipeline(pipeline=pipe)

# Prompt
prompt = PromptTemplate.from_template(
    """You are a financial analyst.

Answer clearly based only on the context.

Context:
{context}

Question:
{question}

Answer:"""
)

# RAG function
def rag_chain(question):
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({"context": context, "question": question})

# Input
query = st.text_input("🔍 Enter your question here:")

# Output
if query:
    with st.spinner("Analyzing..."):
        answer = rag_chain(query)
    st.success(answer)