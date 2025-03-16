import os
import faiss
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load smaller models for local machine
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

# Use a lightweight open-access model
model_name = "HuggingFaceH4/zephyr-7b-alpha"
llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True).to(device)

# Ensure financial data directory exists
if not os.path.exists("./financials"):
    os.makedirs("./financials")

# Load financial documents with flexible encoding handling
def load_documents(folder="./financials"):
    docs = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                docs.append(f.read())
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='ISO-8859-1') as f:
                docs.append(f.read())
    return docs

# Preprocess and embed documents
def embed_documents(docs):
    chunks = [chunk.strip() for doc in docs for chunk in doc.split("\n") if chunk.strip()]

    if not chunks:
        raise ValueError("No valid chunks found in documents.")

    embeddings = embed_model.encode(chunks, convert_to_numpy=True)

    if len(embeddings) == 0:
        raise ValueError("Embedding generation failed. Ensure documents are not empty.")

    # Store in FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))  # Ensure dtype is correct

    # Initialize BM25
    bm25 = BM25Okapi([chunk.split() for chunk in chunks])
    return chunks, index, bm25

# Input Guardrail: Validate financial queries
def is_valid_query(query):
    financial_keywords = ["revenue", "profit", "expenses", "earnings", "cash flow", "balance sheet"]
    return any(keyword in query.lower() for keyword in financial_keywords)

# Hybrid Search (FAISS + BM25)
def hybrid_search(query, index, bm25, chunks, top_k=5):
    query_vec = embed_model.encode([query], convert_to_numpy=True)

    # FAISS dense search
    _, faiss_results = index.search(query_vec.astype(np.float32), top_k)

    # BM25 keyword search
    bm25_results = bm25.get_top_n(query.split(), chunks, n=top_k)

    results = list(set([chunks[i] for i in faiss_results[0] if i < len(chunks)] + bm25_results))
    return results[:top_k]

# Generate response using LLM
def generate_response(context, query):
    prompt = f"Answer the following financial question based on the context below:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
    input_ids = llm_tokenizer.encode(prompt, return_tensors="pt").to(device)

    output = llm.generate(input_ids, max_new_tokens=150, temperature=0.7)
    return llm_tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit UI
def main():
    st.title("Financial RAG Chatbot")

    st.sidebar.header("Instructions")
    st.sidebar.write("Place financial documents in the './financials' folder.")

    # Load and embed documents
    st.sidebar.write("Loading financial documents...")
    documents = load_documents()
    try:
        chunks, index, bm25 = embed_documents(documents)
        st.sidebar.success("Documents loaded successfully!")
    except ValueError as e:
        st.sidebar.error(f"Error: {e}")
        return

    user_query = st.text_input("Ask a financial question:")

    if user_query:
        if not is_valid_query(user_query):
            st.warning("Invalid query: Please ask a relevant financial question.")
            return

        # Retrieve and generate response
        context = "\n".join(hybrid_search(user_query, index, bm25, chunks))
        response = generate_response(context, user_query)

        st.subheader("Answer:")
        st.write(response)

if __name__ == '__main__':
    main()