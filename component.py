from flask import jsonify,request
from PyPDF2 import PdfReader
import re
import time
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import pinecone
from typing import List, Tuple
import os
from dotenv import load_dotenv
# Initialize Pinecone
from pinecone import Pinecone, ServerlessSpec
from groq_llm import generate_question
from groq_llm import generate_answer
from groq_llm import chat_get_answer_to_query

#API Endpoint 1
def delete_index_pinecone(index_name):
    pinecone = Pinecone(api_key="f9ccd300-0b10-436b-8821-cbe8103ee591")
    indlist=[index["name"] for index in pinecone.list_indexes().indexes]
    print(indlist[0])
    print(index_name)
    if any(index["name"] == index_name for index in pinecone.list_indexes().indexes):
        pinecone.delete_index(index_name)
        return jsonify({
            'message': f"{index_name} index has been deleted from Pinecone."
        })
            
    return jsonify({'error': f"Index '{index_name}' does not exist"}), 404
    
    

def create_pinecone_connection(index_name):
    pinecone = Pinecone(api_key="f9ccd300-0b10-436b-8821-cbe8103ee591")
    index_name = index_name
    # Create Pinecone index if it doesn't exist
    if len(pinecone.list_indexes()) == 0:
        pinecone.create_index(
        name=index_name,
        dimension=384, # Replace with your model dimensions it should be same as embedding model dimension
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
        )
    else:
        if any(index["name"] == index_name for index in pinecone.list_indexes().indexes):
            pass
        else:
            pinecone.create_index(
            name=index_name,
            dimension=384, # Replace with your model dimensions
            metric="cosine", # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
            )
            # Connect to the index
    index = pinecone.Index(index_name)
    return index

def extract_text_from_pdf(pdf_file: str) -> str:
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\d+(?=\n)', '', text)
    return text.strip()

def semantic_chunking(text: str, max_chunk_size: int = 512, overlap: int = 50) -> List[str]:
    # Download necessary NLTK data
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
    model.max_seq_length = 512  # Set the maximum sequence length

    embeddings = []
    for chunk in chunks:
        # Split the chunk into smaller parts if it's too long
        if len(model.tokenize(chunk)['input_ids']) > 512:
            sub_chunks = semantic_chunking(chunk, max_chunk_size=512, overlap=50)
            chunk_embeddings = model.encode(sub_chunks)
            # Take the mean of sub-chunk embeddings
            embedding = np.mean(chunk_embeddings, axis=0)
        else:
            embedding = model.encode([chunk])[0]
        embeddings.append(embedding)

    return np.array(embeddings)

def store_embeddings_in_pinecone(indexName:str,chunks: List[str], embeddings: np.ndarray):
    pinecone = Pinecone(api_key="f9ccd300-0b10-436b-8821-cbe8103ee591")
    index = pinecone.Index(indexName)
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector = {
            'id': f'chunk_{i}',
            'values': embedding.tolist(),
            'metadata': {'text': chunk}
        }
        vectors.append(vector)

    # Upsert vectors in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)

#API Endpoint 4
#store the data in pinecone
def store_in_pinecone(indexName:str,pdf_path: str):
    text = extract_text_from_pdf(pdf_path)
    clean_text_content = clean_text(text)
    chunks = semantic_chunking(clean_text_content)
    
    create_pinecone_connection(indexName)
    embeddings = generate_embeddings(chunks)
    store_embeddings_in_pinecone(indexName,chunks, embeddings)

#API Endpoint 3
def get_answer_to_query(query: str,indexName:str, k: int=3):
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY')
    pinecone = Pinecone(api_key="f9ccd300-0b10-436b-8821-cbe8103ee591")
    index = pinecone.Index(indexName)

    # Initialize the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create a query vector
    query_text = query

    query_vector = model.encode([query_text])[0].tolist()

    # Query Pinecone 
    results = index.query(
      vector=query_vector,
      top_k=k,
      include_metadata=True
    )
    similar_chunks=[match['metadata']['text'] for match in results['matches']]
    context = " ".join([chunk for chunk in similar_chunks])
    # print(context)
    answer = generate_answer(query_text, context,api_key)
    return answer
def get_similar_chunks_from_pinecone(query: str,indexName:str, k: int=3):
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY')
    pinecone = Pinecone(api_key="f9ccd300-0b10-436b-8821-cbe8103ee591")
    index = pinecone.Index(indexName)

    # Initialize the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create a query vector
    query_text = query

    query_vector = model.encode([query_text])[0].tolist()

    # Query Pinecone 
    results = index.query(
      vector=query_vector,
      top_k=k,
      include_metadata=True
    )
    similar_chunks=[match['metadata']['text'] for match in results['matches']]
    
    return similar_chunks

def get_random_chunk_from_pinecone(indexName:str,num_chunks: int) -> List[Tuple[float, str]]:
    pinecone = Pinecone(api_key="f9ccd300-0b10-436b-8821-cbe8103ee591")
    index = pinecone.Index(indexName)
    
    index_stats = index.describe_index_stats()
    total_vectors = index_stats.total_vector_count

    # Generate random IDs
    random_ids = np.random.choice(total_vectors, num_chunks, replace=False)

    # Fetch these random vectors
    random_chunks = []
    for id in random_ids:
        id = "chunk_"+str(id)
        try:
            vector = index.fetch(ids=[id])
            
            if vector['vectors']:
                random_chunks.append(vector['vectors'][id]['metadata']['text'])
        except Exception as e:
            print(f"Error fetching vector {id}: {e}")

    return random_chunks


#API Endpoint 2
def generate_flashcards(index_name:str,num_cards: int) -> List[Tuple[str, str]]:

    chunks=get_random_chunk_from_pinecone(index_name,num_cards)
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY')
    flashcards = []
    for i in range(num_cards):
        chunk = chunks[i]
        question = generate_question(chunk,api_key)
        similar_chunks = get_similar_chunks_from_pinecone(question,index_name, k=3)
        context = " ".join([chunk for chunk in similar_chunks])
        answer = generate_answer(question, context,api_key)
        flashcards.append((question, answer))

    return flashcards

#manage conversation history
def manage_conversation_history(query: str, response: str, history: List[dict], max_history: int = 5) -> List[dict]:
    history.append({"query": query, "response": response})
    if len(history) > max_history:
        history = history[-max_history:]
    return history

#Clean up session
def cleanup_old_sessions(conversation_histories,last_access_times,SESSION_TIMEOUT):
    current_time = time.time()
    expired_sessions = [
        session_id for session_id, last_access in last_access_times.items()
        if current_time - last_access > SESSION_TIMEOUT
    ]
    for session_id in expired_sessions:
        del conversation_histories[session_id]
        del last_access_times[session_id]

def chat_get_full_answer_to_query(query:str,indexName: str,formatted_chat_history,k: int=5):
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY')
    pinecone = Pinecone(api_key="f9ccd300-0b10-436b-8821-cbe8103ee591")
    index = pinecone.Index(indexName)

    # Initialize the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create a query vector
    query_text = query

    query_vector = model.encode([query_text])[0].tolist()

    # Query Pinecone 
    results = index.query(
      vector=query_vector,
      top_k=k,
      include_metadata=True
    )
    similar_chunks=[match['metadata']['text'] for match in results['matches']]
    context = " ".join([chunk for chunk in similar_chunks])
    # print(context)
    answer = chat_get_answer_to_query(query_text, context,formatted_chat_history,api_key)
    return answer

