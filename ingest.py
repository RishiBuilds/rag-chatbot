import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DOCS_PATH = os.getenv("DOCS_PATH", "CloudExecX report.pdf")
INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

def main():
    if not os.path.exists(DOCS_PATH):
        print(f"Error: Document not found at '{DOCS_PATH}'.")
        print("Please place the PDF file in the correct directory or update 'DOCS_PATH' in your .env file.")
        sys.exit(1)

    print(f"Loading document: {DOCS_PATH}...")
    try:
        loader = PyPDFLoader(DOCS_PATH)
        documents = loader.load()
    except Exception as e:
        print(f"Error loading PDF document: {e}")
        sys.exit(1)

    print(f"Loaded {len(documents)} pages. Splitting into chunks (size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks.")

    print(f"Loading embedding model: {EMBED_MODEL}...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        sys.exit(1)

    print("Building FAISS vector index...")
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_PATH)
        print(f"Vector store created and saved successfully to '{INDEX_PATH}'!")
    except Exception as e:
        print(f"Error saving vector store: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()