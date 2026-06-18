import os
import sys
import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

warnings.filterwarnings("ignore")

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
DOCS_PATH = os.getenv("DOCS_PATH", "CloudExecX report.pdf")
INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

if not api_key:
    print("Error: 'GROQ_API_KEY' is not set in the environment or .env file.")
    print("Please set it in your .env file, e.g.:")
    print("GROQ_API_KEY=your_groq_api_key_here")
    sys.exit(1)

print(f"Loading embedding model: {EMBED_MODEL}...")
try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
except Exception as e:
    print(f"Error loading embedding model: {e}")
    sys.exit(1)

if os.path.exists(INDEX_PATH):
    print(f"Loading existing vector store from '{INDEX_PATH}'...")
    try:
        vectorstore = FAISS.load_local(
            INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Error loading vector store: {e}")
        sys.exit(1)
else:
    print(f"Vector store not found at '{INDEX_PATH}'. Creating new index...")
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

    print(f"Splitting document into chunks (size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print("Building FAISS vector index...")
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_PATH)
        print(f"Vector store created and saved successfully to '{INDEX_PATH}'!")
    except Exception as e:
        print(f"Error creating/saving vector store: {e}")
        sys.exit(1)

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

prompt = PromptTemplate.from_template("""
You are a helpful assistant answering questions about a specific document.
Use the context below to answer the question.
If the answer isn't in the context, say "I don't know" - don't make things up.

Context: {context}

Question: {question}
Answer:""")

print(f"Initializing LLM (model: {MODEL})...")
try:
    llm = ChatGroq(api_key=api_key, model=MODEL, temperature=0)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
except Exception as e:
    print(f"Error initializing language model chain: {e}")
    sys.exit(1)

GREETINGS = {"hi", "hello", "hey", "hyy", "yo", "hii", "hola"}

def is_greeting(text):
    cleaned = text.lower().strip().strip("!?.")
    return cleaned in GREETINGS

def main():
    print("\n" + "="*65)
    print("Chatbot ready. Ask a question about the document.")
    print("Conversational history is tracked. Type 'exit' to quit.")
    print("="*65)
    
    chat_history = []
    
    while True:
        try:
            query = input("\nYou: ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        if not query:
            continue
        if query.lower() == "exit":
            break
        if is_greeting(query):
            print("\nBot: Hey! Ask me anything about the document.")
            continue
        
        try:
            print("Thinking...")
            result = qa_chain.invoke({"question": query, "chat_history": chat_history})
            
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            
            print(f"\nBot: {answer}")

            if source_docs:
                pages = set()
                for doc in source_docs:
                    page_num = doc.metadata.get("page")
                    if page_num is not None:
                        pages.add(int(page_num) + 1)
                
                if pages:
                    sorted_pages = sorted(list(pages))
                    pages_str = ", ".join(f"Page {p}" for p in sorted_pages)
                    print(f" [Sources: {pages_str}]")

            chat_history.append((query, answer))
            
        except Exception as e:
            print(f"\nError processing your request: {e}")

if __name__ == "__main__":
    main()