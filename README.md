AI PDF Chatbot (RAG + FAISS + Sentence Transformers)
Overview

This project is an AI-powered document question answering system built using Retrieval-Augmented Generation (RAG). Users can upload a PDF document and ask natural language questions. The system retrieves the most relevant sections from the document using semantic search and returns context-based answers.

The application is designed to function as a lightweight document intelligence assistant that can be used for study materials, research papers, notes, and technical documents.

Key Features
Upload and process PDF documents
Automatic text extraction from multi-page PDFs
Chunk-based document processing for efficient retrieval
Semantic search using Sentence Transformers
Fast similarity search using FAISS vector database
Context-based answer generation from retrieved chunks
Interactive web interface using Streamlit
System Architecture

The system follows a Retrieval-Augmented Generation pipeline:

PDF Document Upload
Text Extraction using PyPDF
Text Chunking into smaller segments
Embedding Generation using Sentence Transformers
Vector Storage using FAISS
Query Embedding and Similarity Search
Retrieval of Top Matching Chunks
Display of Contextual Answer
Tech Stack
Python
Streamlit
SentenceTransformers (all-MiniLM-L6-v2)
FAISS (Facebook AI Similarity Search)
PyPDF
NumPy
Project Structure
rag-chatbot/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
└── sample_docs/          # Optional sample PDFs
Installation

Clone the repository and install dependencies:

git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt
Requirements
streamlit
sentence-transformers
faiss-cpu
numpy
pypdf
Running the Application

Start the Streamlit app using:

streamlit run app.py
How It Works
Document Processing

The uploaded PDF is read page by page and converted into raw text. The text is then cleaned and split into smaller chunks to improve retrieval accuracy.

Embedding Generation

Each text chunk is converted into a high-dimensional vector using a pre-trained Sentence Transformer model.

Vector Search

FAISS is used to store and search embeddings efficiently. When a user enters a query, it is also converted into a vector and matched against stored document embeddings.

Response Generation

The most relevant chunks are retrieved and displayed as context-based answers.

Use Cases
Question answering over study notes
Research paper summarization
Document search assistant
Personal knowledge base chatbot
Resume or policy document analyzer
Limitations
Works best with text-based PDFs
Performance depends on chunk quality
No external knowledge beyond uploaded document
No advanced LLM reasoning layer included
Future Improvements
Integration with large language models for better response generation
Multi-document support
Chat history memory
Advanced chunking strategies (semantic splitting)
Reranking model for improved retrieval accuracy
Author

Developed by: Your Name
Domain: Machine Learning, NLP, Retrieval-Augmented Systems
