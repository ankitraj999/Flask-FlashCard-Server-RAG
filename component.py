# component.py

import io
from flask import jsonify
from PyPDF2 import PdfReader
import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import pinecone
from typing import List, Tuple
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from groq_llm import generate_question, generate_answer

# Load environment variables
load_dotenv()

# Initialize Pinecone
pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

PINECONE_INDEX_NAME = "flashcard-embeddings"  # Use a single, consistent index name

def get_or_create_pinecone_index():
    pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    
    try:
        # Try to get the existing index
        return pinecone.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        # If the index doesn't exist, create it
        if "index not found" in str(e).lower():
            try:
                pinecone.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=384,  # Adjust based on your embedding model
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                print(f"Index '{PINECONE_INDEX_NAME}' created successfully.")
                return pinecone.Index(PINECONE_INDEX_NAME)
            except Exception as create_error:
                if "ALREADY_EXISTS" in str(create_error):
                    # If we get here, the index was created by another request
                    print(f"Index '{PINECONE_INDEX_NAME}' already exists. Using existing index.")
                    return pinecone.Index(PINECONE_INDEX_NAME)
                else:
                    raise
        else:
            raise
def store_embeddings_in_pinecone(index, chunks: List[str], embeddings: np.ndarray):
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector = {
            'id': f'chunk_{i}',
            'values': embedding.tolist(),
            'metadata': {'text': chunk}
        }
        vectors.append(vector)

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)

def extract_text_from_pdf(pdf_file: io.BytesIO) -> str:
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    print(f"No text extracted from page {page_num + 1}. It might be blank or contain only images.")
            except Exception as e:
                print(f"Error extracting text from page {page_num + 1}: {str(e)}")
                # Continue to the next page instead of breaking the loop
                continue
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF. The file might be empty, contain only images, or be encrypted.")
    except Exception as e:
        raise ValueError(f"Error reading PDF: {str(e)}")
    
    return text

# Update the store_in_pinecone function to use the new extract_text_from_pdf function

def store_in_pinecone(class_number: str, book_name: str, chapter_number: str, pdf_file: io.BytesIO):
    try:
        text = extract_text_from_pdf(pdf_file)
        if not text:
            raise ValueError("No text could be extracted from the PDF.")
        
        clean_text_content = clean_text(text)
        chunks = semantic_chunking(clean_text_content)
        embeddings = generate_embeddings(chunks)
        
        index = get_or_create_pinecone_index()
        
        # Add class, book, and chapter information to the metadata
        vectors = [
            {
                'id': f"class_{class_number}_book_{book_name}_chapter_{chapter_number}_chunk_{i}",
                'values': embedding.tolist(),
                'metadata': {
                    'text': chunk,
                    'class': class_number,
                    'book': book_name,
                    'chapter': chapter_number
                }
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        
        # Upsert vectors to the index
        index.upsert(vectors=vectors)
        
        return "Text extracted and stored successfully"
    except ValueError as e:
        raise
    except Exception as e:
        print(f"Unexpected error in store_in_pinecone: {str(e)}")
        raise ValueError(f"An unexpected error occurred while processing and storing the PDF content: {str(e)}")

    
def delete_index_pinecone(index_name: str):
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
        return jsonify({
            'message': f"{index_name} index has been deleted from Pinecone."
        })
    return jsonify({'error': f"Index '{index_name}' does not exist"}), 404

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\d+(?=\n)', '', text)
    return text.strip()

def semantic_chunking(text: str, max_chunk_size: int = 512, overlap: int = 50) -> List[str]:
    nltk.download('punkt', quiet=True)
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-1:]
            current_size = len(current_chunk[0])

        current_chunk.append(sentence)
        current_size += sentence_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def generate_embeddings(chunks: List[str]) -> np.ndarray:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.max_seq_length = 512

    embeddings = []
    for chunk in chunks:
        if len(model.tokenize(chunk)['input_ids']) > 512:
            sub_chunks = semantic_chunking(chunk, max_chunk_size=512, overlap=50)
            chunk_embeddings = model.encode(sub_chunks)
            embedding = np.mean(chunk_embeddings, axis=0)
        else:
            embedding = model.encode([chunk])[0]
        embeddings.append(embedding)

    return np.array(embeddings)

def get_answer_to_query(query: str, index_name: str, k: int = 3):
    index = pinecone.Index(index_name)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vector = model.encode([query])[0].tolist()

    results = index.query(vector=query_vector, top_k=k, include_metadata=True)
    similar_chunks = [match['metadata']['text'] for match in results['matches']]
    context = " ".join(similar_chunks)

    api_key = os.getenv('GROQ_API_KEY')
    answer = generate_answer(query, context, api_key)
    return answer

def generate_flashcards(class_number: str, book_name: str, chapter_number: str, num_cards: int = 10):
    index = get_or_create_pinecone_index()
    
    # Query vectors for the specific class, book, and chapter
    query_response = index.query(
        vector=[0] * 384,  # Dummy vector, we're not using similarity search here
        filter={
            "class": {"$eq": class_number},
            "book": {"$eq": book_name},
            "chapter": {"$eq": chapter_number}
        },
        top_k=num_cards,
        include_metadata=True
    )
    
    flashcards = []
    api_key = os.getenv('GROQ_API_KEY')
    for match in query_response.matches:
        chunk = match.metadata['text']
        question = generate_question(chunk, api_key)
        answer = generate_answer(question, chunk, api_key)
        flashcards.append((question, answer))


    
    return flashcards
def get_answer_to_query(query: str, class_number: str, book_name: str, chapter_number: str, k: int = 3):
    pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pinecone.Index(PINECONE_INDEX_NAME)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vector = model.encode([query])[0].tolist()

    results = index.query(
        vector=query_vector,
        filter={
            "class": {"$eq": class_number},
            "book": {"$eq": book_name},
            "chapter": {"$eq": chapter_number}
        },
        top_k=k,
        include_metadata=True
    )
    
    similar_chunks = [match.metadata['text'] for match in results.matches]
    context = " ".join(similar_chunks)

    api_key = os.getenv('GROQ_API_KEY')
    answer = generate_answer(query, context, api_key)
    return answer
