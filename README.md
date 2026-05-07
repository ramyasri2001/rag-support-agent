# RAG Customer Support Agent 🤖

A production-grade RAG (Retrieval-Augmented Generation) 
customer support agent built with LangChain, Groq LLM API, 
ChromaDB vector database, and Flask REST API.

## 🏗️ Architecture
User Question
↓
Flask REST API
↓
LangChain Orchestration
↓
ChromaDB Semantic Search
↓
Groq LLM (Llama 3.1)
↓
Accurate Answer! 
## 🛠️ Tech Stack

- **LangChain** — AI orchestration framework
- **Groq LLM API** — Llama 3.1 language model
- **ChromaDB** — Vector database for semantic search
- **Flask** — REST API framework
- **Python** — Core programming language
- **Sentence Transformers** — Text embeddings

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Groq API key (free at groq.com)

### Installation

```bash
# Clone the repository
git clone https://github.com/ramyasri2001/rag-support-agent.git
cd rag-support-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install langchain langchain-groq langchain-community chromadb flask python-dotenv pypdf sentence-transformers
```

### Configuration

```bash
# Create .env file
touch .env

# Add your Groq API key
echo "GROQ_API_KEY=your_groq_api_key_here" >> .env
```

### Run the Application

```bash
python3 app.py
```

Server starts at: `http://localhost:8080`

## 📡 API Endpoints

### Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "model": "llama-3.1-8b-instant",
  "vectorstore": "ChromaDB"
}
```

### Ask a Question
```bash
POST /ask
Content-Type: application/json

{
  "question": "What is your return policy?"
}

Response:
{
  "answer": "Our return policy allows returns within 30 days...",
  "question": "What is your return policy?",
  "status": "success"
}
```

## 💬 Example Queries

```bash
# Return Policy
curl -X POST http://localhost:8080/ask \
-H "Content-Type: application/json" \
-d '{"question": "What is your return policy?"}'

# Shipping
curl -X POST http://localhost:8080/ask \
-H "Content-Type: application/json" \
-d '{"question": "How long does shipping take?"}'

# Password Reset
curl -X POST http://localhost:8080/ask \
-H "Content-Type: application/json" \
-d '{"question": "How do I reset my password?"}'

# Customer Support
curl -X POST http://localhost:8080/ask \
-H "Content-Type: application/json" \
-d '{"question": "How can I contact customer support?"}'
```

## 🔑 Key Features

- ✅ Semantic search using vector embeddings
- ✅ Context-aware responses using RAG architecture
- ✅ Production-grade Flask REST API
- ✅ ChromaDB persistent vector storage
- ✅ Groq LLM for fast inference
- ✅ Easy to extend with new knowledge base documents

## 📁 Project Structure
rag-support-agent/
├── app.py              # Main Flask application
├── .env                # Environment variables (not committed)
├── .gitignore          # Git ignore file
├── chroma_db/          # ChromaDB vector store
└── README.md           # Project documentation
## 🧠 How RAG Works

1. **Indexing**: Documents are split into chunks and 
   converted to vector embeddings using Sentence Transformers
2. **Storage**: Embeddings stored in ChromaDB vector database
3. **Retrieval**: User question converted to embedding,
   semantically similar chunks retrieved from ChromaDB
4. **Generation**: Retrieved context + question sent to 
   Groq LLM to generate accurate, grounded response

## 👩‍💻 Author

**Ramyasri Kanugula**
- MS Computer Science + MBA
- GitHub: [@ramyasri2001](https://github.com/ramyasri2001)
- LinkedIn: [ramyasri-kanugula](https://linkedin.com/in/ramyasri-kanugula-763012210)
