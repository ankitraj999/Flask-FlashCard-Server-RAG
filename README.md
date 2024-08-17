# PDF Chat and Flashcard Generator

This Flask application provides a backend service for chatting with PDF files and generating flashcards using Retrieval-Augmented Generation (RAG). It utilizes Pinecone for vector storage and NLTK for text chunking.

## Features

- Upload and process PDF files
- Store embeddings in Pinecone vector database
- Generate flashcards from PDF content
- Answer user queries based on PDF content
- Delete indexes from Pinecone database

## Endpoints

1. **Delete Index**
   - `DELETE /api/flashcard/index`
   - Deletes a specified index from the Pinecone vector database.

2. **Upload PDF**
   - `POST /api/flashcard/upload`
   - Uploads a PDF file, processes its content, and stores embeddings in Pinecone.

3. **Generate Flashcards**
   - `GET /api/flashcard/create`
   - Generates a random number of flashcards based on the uploaded PDF content.

4. **Answer Query**
   - `GET /api/flashcard/query`
   - Answers user queries using the RAG model based on the uploaded PDF content.

## Technologies Used

- Flask: Web framework for the backend
- Pinecone: Vector database for storing embeddings
- NLTK: Natural Language Toolkit for text processing and chunking
- PyPDF2: PDF processing library
- Transformers: For generating embeddings and implementing RAG

## Setup and Installation

1. Clone the repository:
