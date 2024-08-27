from groq import Groq
import os
from dotenv import load_dotenv
import re

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

def chat_get_answer_to_query(current_question: str, context: str,chat_history,api_key) -> str:
    # api_key=API_KEY
    client = Groq(api_key=api_key)
    prompt_template = """
    You are an AI assistant tasked with answering questions based on specific provided context. Your role is to understand the user's question, analyze the given context, and provide an accurate response derived solely from that context.

    Context:
    {context}

    Chat history:
    {chat_history}

    User's current question:
    {current_question}

    Instructions:
    1. Carefully read and understand the provided context.
    2. Review the chat history to understand the conversation flow.
    3. Analyze the user's current question.
    4. Formulate your response based ONLY on the information given in the context.
    5. If the answer cannot be fully derived from the context, state this clearly and explain what information is missing.
    6. Maintain consistency with any information shared in previous exchanges.
    7. If the question is unclear or ambiguous, ask for clarification before answering.

    Your response should be:
    1. Directly relevant to the current question
    2. Based solely on the provided context
    3. Consistent with the conversation history
    4. Clear, concise, and informative

    Do not use any external knowledge or information not present in the given context. If the context doesn't contain the necessary information to answer the question, state this explicitly.

    Please provide your response only do not include any other things.Just give simple response.
    """
    
    formatted_chat_history = format_chat_history(chat_history)
    full_prompt = prompt_template.format(
        context=context,
        chat_history=formatted_chat_history,
        current_question=current_question
    )
    message=[
        {
            "role": "user",
            "content":full_prompt
        },
        {
            "role": "system",
            "content": "You are an AI assistant tasked with answering questions based on specific provided context. Your role is to understand the user's question, analyze the given context, and provide an accurate response derived solely from that context."
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
def format_chat_history(chat_history):
    formatted_history = ""
    for entry in chat_history:
        formatted_history += f"Human: {entry['query']}\n"
        formatted_history += f"Assistant: {entry['response']}\n\n"
    return formatted_history
