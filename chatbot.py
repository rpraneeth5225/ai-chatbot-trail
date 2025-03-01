import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama  # Local LLM

# Set Streamlit Page Configuration
st.set_page_config(page_title="PDF Chatbot", layout="wide")

# Title
st.title("📄💬 Chat with Your PDF")

# Sidebar for File Upload
with st.sidebar:
    st.header("📂 Upload Your Document")
    uploaded_file = st.file_uploader("Upload a PDF file to begin", type=["pdf"])

# If a file is uploaded
if uploaded_file:
    # Read the PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    # Display file name
    st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(text)

    # Generate embeddings
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User Input for Questions
    user_question = st.text_input("❓ Ask a question about the PDF:")

    if user_question:
        with st.spinner("Thinking... 🤔"):
            # Perform similarity search
            matched_docs = vector_store.similarity_search(user_question)

            # Load Local LLM (Ollama) and Run Query
            llm = Ollama(model="mistral")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=matched_docs, question=user_question)

            # Display Response
            st.subheader("🧠 AI Response:")
            st.write(response)
