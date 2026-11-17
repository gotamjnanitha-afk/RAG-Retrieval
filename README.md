
# Understanding RAG Retrieval

## Project Overview

This project demonstrates Retrieval-Augmented Generation (RAG) using Python.

Users can upload PDF documents and ask questions. The application retrieves relevant information from the uploaded document and generates answers using an AI language model.

## Technologies Used

- Python
- Streamlit
- LangChain
- Hugging Face Embeddings
- FAISS Vector Database
- Groq LLM
- PyPDF

## Working Process

1. Upload a PDF document.
2. Extract text from the PDF.
3. Split the text into smaller chunks.
4. Generate embeddings using MiniLM.
5. Store embeddings in FAISS.
6. Retrieve relevant chunks using similarity search.
7. Send the retrieved context to the Groq language model.
8. Generate an answer based on the document.

## Installation

Clone the repository:

```bash
git clone https://github.com/gotamjnanitha-afk/RAG-Retrieval.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your Groq API key:

```text
GROQ_API_KEY=your_groq_api_key
```

Run the application:

```bash
streamlit run app.py
```

## Features

- PDF document upload
- Automatic text extraction
- Text chunking
- Embedding generation
- FAISS similarity search
- AI-powered question answering
- Retrieved information display

## Project Objective

To understand how RAG retrieves relevant information from external documents and uses it to generate context-based answers.

## Author

Jnanitha
