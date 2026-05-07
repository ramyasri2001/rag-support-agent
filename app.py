from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.1
)

# Initialize embeddings
embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Sample knowledge base
# In production this would be
# loaded from real documents!
documents = [
    Document(page_content="""
    Our return policy allows returns 
    within 30 days of purchase. 
    Items must be unused and in 
    original packaging. Refunds are 
    processed within 5-7 business days.
    """),
    Document(page_content="""
    Shipping takes 3-5 business days 
    for standard delivery. Express 
    shipping takes 1-2 business days. 
    Free shipping on orders over $50.
    """),
    Document(page_content="""
    To reset your password, click 
    'Forgot Password' on the login page. 
    Enter your email address and we will 
    send a reset link within 5 minutes.
    """),
    Document(page_content="""
    Customer support is available 
    Monday to Friday, 9 AM to 6 PM EST. 
    Contact us at support@company.com 
    or call 1-800-123-4567.
    """),
    Document(page_content="""
    Our premium membership costs $9.99 
    per month or $99 per year. Benefits 
    include free shipping, early access 
    to sales, and priority support.
    """),
]

# Create vector store
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split documents
splits = text_splitter.split_documents(documents)

# Create ChromaDB vector store
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "RAG Customer Support Agent",
        "status": "running",
        "endpoints": {
            "ask": "/ask (POST)",
            "health": "/health (GET)"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": "llama3-8b-8192",
        "vectorstore": "ChromaDB"
    })

@app.route("/ask", methods=["POST"])
def ask():
    try:
        # Get question from request
        data = request.get_json()
        
        if not data or "question" not in data:
            return jsonify({
                "error": "Please provide a question!"
            }), 400
        
        question = data["question"]
        
        # Get answer from RAG chain
        result = qa_chain({"query": question})
        
        answer = result["result"]
        
        return jsonify({
            "question": question,
            "answer": answer,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == "__main__":
    print("Starting RAG Support Agent...")
    print("Visit http://localhost:5000")
    app.run(debug=True, port=8080)