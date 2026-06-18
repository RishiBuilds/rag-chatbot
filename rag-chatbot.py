import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.1-8b-instant"
DOCS_PATH = "CloudExecX report.pdf"
INDEX_PATH = "faiss_index"
TOP_K = 4
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GREETINGS = {"hi", "hello", "hey", "hyy", "yo", "hii", "hola"}

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

if os.path.exists(INDEX_PATH):
    vectorstore = FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
else:
    loader = PyPDFLoader(DOCS_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

prompt = PromptTemplate.from_template("""
You are a helpful assistant answering questions about a specific document.
Use the context below to answer the question.
If the answer isn't in the context, say "I don't know" - don't make things up.

Context: {context}
Question: {question}
Answer:""")

llm = ChatGroq(api_key=api_key, model=MODEL, temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
)


def is_greeting(text):
    cleaned = text.lower().strip().strip("!?.")
    return cleaned in GREETINGS


def main():
    print("Chatbot ready. Ask a question about the document, or type 'exit' to quit.")
    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break
        if is_greeting(query):
            print("\nBot: Hey! Ask me anything about the document.")
            continue
        result = qa_chain.invoke({"query": query})
        print(f"\nBot: {result['result']}")


if __name__ == "__main__":
    main()