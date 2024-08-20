from groq import Groq
import os
from dotenv import load_dotenv
import re
load_dotenv()

def generate_question(chunk,api_key) -> str:
   
    client = Groq(api_key=api_key)

    message = [
    {
        "role": "system",
        "content": "You are an expert educator specializing in creating effective flashcards. Your task is to generate clear, concise, and thought-provoking questions based on the given text. Follow these guidelines:\n\n1. Focus on key concepts, important facts, or central ideas.\n2. Avoid yes/no questions; prefer open-ended or specific answer questions.\n3. Use clear and concise language.\n4. Ensure the question can be answered based solely on the provided text.\n5. Aim for questions that test understanding, not just memorization.\n6. Do not include the answer in your response, only the question."
    },
    {
        "role": "user",
        "content": f"Based on the following text, generate one high-quality short answer flashcard question:\n\n{chunk}. Give only question just like when someone ask question and you answer "
    }
]
    try:
        completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=message,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
        )

        result=""
        for chunk in completion:
            result+=chunk.choices[0].delta.content or ""
    
        # result=re.sub(r'\s+', ' ', result).strip()
        return result
    except Exception as e:
        return f"An error occured: {e}"

def generate_answer(question: str, context: str,api_key) -> str:
    # api_key=API_KEY
    client = Groq(api_key=api_key)

    message=[
        {
            "role": "user",
            "content":f"Question: {question}"
        },
        {
            "role": "system",
            "content": f"Use the following context to answer the flashcard question: {context}. Give only precise short-answer just like when someone ask question and you answer"
        }
        ]
    try:
        completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=message,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
        )

        result=""
        for chunk in completion:
            result+=chunk.choices[0].delta.content or ""
    
        # result=re.sub(r'\s+', ' ', result).strip()
        return result.strip()
    except Exception as e:
        return f"An error occured: {e}"