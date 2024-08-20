# firebase_config.py

import firebase_admin
from firebase_admin import credentials, firestore

def initialize_firebase():
    cred = credentials.Certificate("pantry-tracker-bee18-firebase-adminsdk-zhwu3-e2e531afa6.json")
    firebase_admin.initialize_app(cred)
    return firestore.client()

def save_flashcards(db, class_number, book_name, chapter_number, flashcards_data):
    doc_ref = db.collection(f'class_{class_number}').document(book_name)
    
    # Check if the document exists
    doc = doc_ref.get()
    if doc.exists:
        # If it exists, update the chapters field
        doc_ref.update({
            f'chapters.chapter_{chapter_number}': flashcards_data
        })
    else:
        # If it doesn't exist, create a new document with the chapters field
        doc_ref.set({
            'chapters': {
                f'chapter_{chapter_number}': flashcards_data
            }
        })

    print(f"Flashcards saved successfully for Class {class_number}, {book_name}, Chapter {chapter_number}")
