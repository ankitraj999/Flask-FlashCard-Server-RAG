from flask import Flask,jsonify,request

from component import delete_index_pinecone
from component import generate_flashcards
from component import get_answer_to_query
from component import store_in_pinecone
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import io
#flask app instance
app=Flask(__name__)
CORS(app)
@app.route("/api/home",methods=['GET'])
def return_home():
    return jsonify({
        'message':"flash card home"
    }), 200

#API Endpoint 1
@app.route("/api/flashcard/index", methods=['DELETE'])
def delete_index():
    index_name = request.args.get('index', '')
    
    if not index_name:
        return jsonify({'error': 'Index name is required'}), 400
    
    try:
        # Assuming pinecone is already initialized
        message=delete_index_pinecone(index_name)
        return message
    except Exception as e:
        return jsonify({
            'error': f"Failed to delete index: {str(e)}"
        }), 500

#API Endpoint 2
@app.route("/api/flashcard/create", methods=['GET'])
def get_flashcards():
    # Get the items from the query parameter
    flashcount = request.args.get('flash_count', '')
    index = request.args.get('index', '')

    if not flashcount:
        return jsonify({'error': 'Flash Card count is required'}), 400
    
    if not index:
        return jsonify({'error': 'Index name is required'}), 400
    
    try:
        # Convert index_name to an integer
        flash_count = int(flashcount)
    except ValueError:
        return jsonify({'error': 'Flash count must be a valid integer'}), 400
    try:
        flashcards=generate_flashcards(index,flash_count)
        flashcards_json = {key: value for key, value in flashcards}
        return jsonify(flashcards_json)
    except Exception as e:
        return jsonify({
            'error': f"Failed to generate Flash cards: {str(e)}"
        }), 500

#API Endpoint 3
@app.route("/api/flashcard/query", methods=['GET'])
def get_flashcards_answer():
    # Get the items from the query parameter
    flashQuery = request.args.get('query', '')
    index = request.args.get('index', '')

    if not flashQuery:
        return jsonify({'error': 'Flash Card query is required'}), 400

    if not index:
        return jsonify({'error': 'Index name is required'}), 400
    
    try:
        answer=get_answer_to_query(flashQuery,index)
        
        return jsonify({
            'answer': answer
        })
    except Exception as e:
        return jsonify({
            'error': f"Failed to generate Flash cards: {str(e)}"
        }), 500

#API Endpoint 4
@app.route('/api/flashcard/upload', methods=['POST'])
def process_pdf():
    index = request.args.get('index', '')
    if not index:
        return jsonify({'error': 'Index name is required'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        try:
            # Read the file into memory
            pdf_file = io.BytesIO(file.read())
            
            # Extract text from the PDF
            extracted_text = store_in_pinecone(index,pdf_file)
            
            return jsonify({
                'message': 'PDF processed successfully'
                
            }), 200
        except Exception as e:
            return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500
    else:
        return jsonify({'error': 'File must be a PDF'}), 400

if __name__=="__main__":
    app.run(debug=True)