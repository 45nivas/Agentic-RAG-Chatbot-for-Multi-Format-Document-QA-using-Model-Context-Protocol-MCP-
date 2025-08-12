from flask import Flask, render_template, request, jsonify, session
import os
import uuid
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
import docx
from pptx import Presentation
import csv
import io

load_dotenv()

app = Flask(__name__)

# Production-ready configuration - Updated for deployment
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB for production
app.config['ENV'] = os.environ.get('FLASK_ENV', 'production')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini AI
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'pptx', 'csv', 'txt', 'md'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(filepath, filename):
    """Extract text content from uploaded files"""
    try:
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        if file_ext == 'pdf':
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
                
        elif file_ext == 'docx':
            doc = docx.Document(filepath)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
            
        elif file_ext == 'pptx':
            prs = Presentation(filepath)
            text = ""
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
            return text
            
        elif file_ext == 'csv':
            with open(filepath, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                rows = list(csv_reader)
                # Convert CSV to readable text format
                text = ""
                for row in rows:
                    text += " | ".join(row) + "\n"
                return text
            
        elif file_ext in ['txt', 'md']:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
                
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {str(e)}")
        return f"Error reading file: {str(e)}"
    
    return "Could not extract text from this file type."

@app.route('/')
def index():
    if 'conversation' not in session:
        session['conversation'] = []
    if 'uploaded_files' not in session:
        session['uploaded_files'] = []
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files selected'}), 400
        
        files = request.files.getlist('files')
        uploaded_files = []
        
        for file in files:
            if file.filename == '':
                continue
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                
                file.save(filepath)
                
                # Extract text content
                text_content = extract_text_from_file(filepath, filename)
                
                uploaded_files.append({
                    'filename': filename,
                    'filepath': filepath,
                    'content': text_content[:2000]  # Store first 2000 chars for session
                })
        
        if not uploaded_files:
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        session['uploaded_files'] = uploaded_files
        session.modified = True
        
        return jsonify({
            'success': True,
            'message': f'{len(uploaded_files)} file(s) uploaded and processed successfully',
            'files': [f['filename'] for f in uploaded_files]
        })
    
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        uploaded_files = session.get('uploaded_files', [])
        if not uploaded_files:
            return jsonify({'error': 'Please upload documents first'}), 400
        
        # Get document content
        document_context = ""
        for file_info in uploaded_files:
            filepath = file_info.get('filepath')
            filename = file_info.get('filename')
            if filepath and os.path.exists(filepath):
                # Re-extract full content for AI processing
                full_content = extract_text_from_file(filepath, filename)
                document_context += f"\n\n=== Content from {filename} ===\n{full_content}\n"
        
        # Generate AI response using Gemini
        if GEMINI_API_KEY and document_context:
            try:
                prompt = f"""You are a helpful AI assistant specializing in document analysis. 
                
Based on the following document content, please answer the user's question:

DOCUMENT CONTENT:
{document_context}

USER QUESTION: {user_message}

Please provide a helpful, accurate response based on the document content. If the information isn't in the documents, please say so."""

                response = model.generate_content(prompt)
                ai_response = response.text
                
            except Exception as e:
                logger.error(f"Gemini API error: {str(e)}")
                ai_response = f"I found your documents about {', '.join([f['filename'] for f in uploaded_files])}, but I'm having trouble processing them right now. The documents appear to contain information, but I cannot provide a detailed analysis at the moment."
        else:
            ai_response = f"I can see you've uploaded {len(uploaded_files)} file(s), but I need the Gemini API to be properly configured to analyze the content and answer your question."
        
        conversation = session.get('conversation', [])
        conversation.append({
            'user': user_message,
            'assistant': ai_response,
            'metadata': {
                'files_processed': len(uploaded_files),
                'has_ai': bool(GEMINI_API_KEY),
                'timestamp': datetime.now().isoformat()
            }
        })
        session['conversation'] = conversation
        session.modified = True
        
        return jsonify({
            'success': True,
            'response': ai_response,
            'metadata': {
                'files_processed': len(uploaded_files),
                'has_ai': bool(GEMINI_API_KEY),
                'timestamp': datetime.now().isoformat()
            }
        })
            
    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}")
        return jsonify({'error': f'Chat processing failed: {str(e)}'}), 500

@app.route('/clear', methods=['POST'])
def clear_conversation():
    try:
        uploaded_files = session.get('uploaded_files', [])
        for file_info in uploaded_files:
            filepath = file_info.get('filepath')
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        
        session['conversation'] = []
        session['uploaded_files'] = []
        session.modified = True
        
        return jsonify({'success': True, 'message': 'Conversation cleared'})
    
    except Exception as e:
        return jsonify({'error': f'Clear failed: {str(e)}'}), 500

@app.route('/health')
def health():
    try:
        has_files = 'uploaded_files' in session and len(session['uploaded_files']) > 0
        return jsonify({
            'status': 'healthy',
            'service': 'Agentic RAG Chatbot - AI Enabled',
            'has_files': has_files,
            'ai_enabled': bool(GEMINI_API_KEY),
            'timestamp': datetime.now().isoformat(),
            'version': '2.1.0'
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    try:
        return app.send_static_file('favicon.ico')
    except:
        return '', 204

@app.route('/health')
def health_check():
    """Health check endpoint for Docker and cloud deployments"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200

if __name__ == '__main__':
    # Production-ready server configuration
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = app.config['DEBUG']
    
    logger.info(f"Starting RAG Chatbot server on {host}:{port}")
    app.run(debug=debug, host=host, port=port)
