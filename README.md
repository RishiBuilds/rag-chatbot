# simple RAG Chatbot

A document Q&A chatbot built entirely with free tools - no OpenAI, no paid vector DB, and no cloud hosting costs.

## Features

- **Conversational Memory**: Remembers prior conversation turns so you can ask follow-up questions.
- **Source Page Citation**: Automatically displays which pages of the PDF document the answers were retrieved from.
- **Flexible Configuration**: Control settings (like the PDF file path, LLM model, chunk sizes, and database names) directly from a `.env` file without modifying source code.

## Stack

- **Embeddings:** `all-MiniLM-L6-v2` (HuggingFace, run locally)
- **Vector store:** FAISS (local, saved on disk)
- **LLM:** Llama 3.1 8B via Groq (free, high-speed API)
- **Framework:** LangChain

## Setup

1. **Clone the repository and set up a virtual environment:**

   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # Mac/Linux
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Copy the example environment template file to create your `.env`:
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and set your `GROQ_API_KEY`. You can get a free, high-rate-limit key at [console.groq.com](https://console.groq.com).

## Usage

You can run the chatbot directly, or pre-ingest the document into the vector database first:

### Option A: Standard Run (Recommended)

1. Place your target PDF document in the project folder.
2. Update the `DOCS_PATH` environment variable in your `.env` file to match your document's file name (defaults to `CloudExecX report.pdf`).
3. Run the chatbot:
   ```bash
   python rag-chatbot.py
   ```
   _Note: If the FAISS database directory (`faiss_index/`) does not exist, `rag-chatbot.py` will automatically build it on startup._

### Option B: Separate Ingestion (Optional)

If you want to build or rebuild the vector database index without starting the chatbot:

```bash
python ingest.py
```

This reads the document, chunks it, creates the embeddings, and saves the FAISS index to the configured directory.

_Note: Subsequent chatbot runs will load the index instantly. To rebuild the index after modifying the source PDF, either delete the index folder or re-run `ingest.py`._

## Notes

- The bot only answers using context from the document - it says "I don't know" rather than guessing.
- `faiss_index/` and `.env` are listed in `.gitignore` to prevent sensitive credentials or local indexes from being committed.
