# index.py

import re
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import io
from component import PINECONE_INDEX_NAME, store_in_pinecone, generate_flashcards, get_answer_to_query, delete_index_pinecone
from flask_cors import CORS
from firbase_config import initialize_firebase, save_flashcards

app = Flask(__name__)
CORS(app)

# Initialize Firebase
db = initialize_firebase()

@app.route("/api/home", methods=['GET'])
def return_home():
    return jsonify({
        'message': "flash card home"
    }), 200

@app.route("/api/flashcard/index", methods=['DELETE'])
def delete_index():
    class_number = request.args.get('class', '')
    book_name = request.args.get('book', '')
    chapter_number = request.args.get('chapter', '')
    
    if not all([class_number, book_name, chapter_number]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        index_name = f"class_{class_number}_{book_name}_chapter_{chapter_number}"
        message = delete_index_pinecone(index_name)
        return message
    except Exception as e:
        return jsonify({
            'error': f"Failed to delete index: {str(e)}"
        }), 500

@app.route("/api/flashcard/create", methods=['GET'])
def get_flashcards():
    flashcount = request.args.get('flash_count', '')
    class_number = request.args.get('class', '')
    book_name = request.args.get('book', '')
    chapter_number = request.args.get('chapter', '')

    if not all([flashcount, class_number, book_name, chapter_number]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        flash_count = int(flashcount)
        index_name = f"class_{class_number}_{book_name}_chapter_{chapter_number}"
        flashcards = generate_flashcards(index_name, flash_count)
        flashcards_json = {key: value for key, value in flashcards}
        
        # Save flashcards to Firebase
        save_flashcards(db, class_number, book_name, chapter_number, flashcards_json)
        
        return jsonify(flashcards_json)
    except Exception as e:
        return jsonify({
            'error': f"Failed to generate Flash cards: {str(e)}"
        }), 500

@app.route("/api/flashcard/query", methods=['GET'])
def get_flashcards_answer():
    flash_query = request.args.get('query', '')
    class_number = request.args.get('class', '')
    book_name = request.args.get('book', '')
    chapter_number = request.args.get('chapter', '')

    if not all([flash_query, class_number, book_name, chapter_number]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        answer = get_answer_to_query(flash_query, class_number, book_name, chapter_number)
        return jsonify({
            'answer': answer
        })
    except Exception as e:
        app.logger.error(f"Error generating answer: {str(e)}")
        return jsonify({
            'error': f"Failed to generate answer: {str(e)}"
        }), 500

@app.route('/api/flashcard/upload', methods=['POST'])
def process_pdf():
    class_number = request.args.get('class', '')
    book_name = request.args.get('book', '')
    chapter_number = request.args.get('chapter', '')

    if not all([class_number, book_name, chapter_number]):
        return jsonify({'error': 'Missing required parameters'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        try:
            pdf_file = io.BytesIO(file.read())
            
            result = store_in_pinecone(class_number, book_name, chapter_number, pdf_file)
            
            if result == "Text extracted and stored successfully":
                # Generate flashcards
                flashcards = generate_flashcards(class_number, book_name, chapter_number, 10)
                flashcards_json = {key: value for key, value in flashcards}
                
                # Save flashcards to Firebase
                save_flashcards(db, class_number, book_name, chapter_number, flashcards_json)
                
                return jsonify({
                    'message': 'PDF processed and flashcards saved successfully',
                    'index_name': PINECONE_INDEX_NAME
                }), 200
            else:
                return jsonify({'error': 'Failed to extract and store text from PDF'}), 400
        except ValueError as e:
            error_message = str(e)
            app.logger.error(f"Error processing PDF: {error_message}")
            return jsonify({'error': error_message}), 400
        except Exception as e:
            app.logger.error(f"Unexpected error processing PDF: {str(e)}")
            return jsonify({'error': 'An unexpected error occurred while processing the PDF. Please try again or contact support if the issue persists.'}), 500
    else:
        return jsonify({'error': 'File must be a PDF'}), 400
    
if __name__ == "__main__":
    app.run(debug=True)