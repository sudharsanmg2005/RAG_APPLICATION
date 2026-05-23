import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

st.set_page_config(page_title="Local PDF RAG App")
st.title("📄 PDF Question Answering with Ollama")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("PDF uploaded successfully!")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Local Ollama model
    llm = Ollama(model="llama3")

    query = st.text_input("Ask a question from the PDF")

    if query:
        with st.spinner("Thinking..."):
            # Retrieve relevant chunks
            retrieved_docs = retriever.invoke(query)

            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            prompt = f"""
Answer the question only from the provided context.

Context:
{context}

Question:
{query}
"""

            response = llm.invoke(prompt)

            st.subheader("Answer")
            st.write(response)

            st.subheader("Source Chunks")
        
            for doc in retrieved_docs:
                st.write(doc.page_content[:500])
                st.write("---")