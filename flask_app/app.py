from flask import Flask, render_template, request, jsonify, session
import os
import sys
import uuid
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import tempfile
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agents.coordinator_agent import CoordinatorAgent

app = Flask(__name__)

# Production-ready configuration
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB for production
app.config['ENV'] = os.environ.get('FLASK_ENV', 'production')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'pptx', 'csv', 'txt', 'md'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
                uploaded_files.append({
                    'filename': filename,
                    'filepath': filepath
                })
        
        if not uploaded_files:
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        session['uploaded_files'] = uploaded_files
        session.modified = True
        
        return jsonify({
            'success': True,
            'message': f'{len(uploaded_files)} file(s) uploaded successfully',
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
        
        coordinator = CoordinatorAgent()
        file_paths = [f['filepath'] for f in uploaded_files]
        
        try:
            messages = coordinator.process(file_paths, user_message)
        except Exception as api_error:
            logger.error(f"API error in coordinator: {str(api_error)}")
            if "API" in str(api_error) or "network" in str(api_error).lower():
                return jsonify({'error': 'Network or API error. Please check your internet connection and try again.'}), 500
            else:
                return jsonify({'error': f'Processing error: {str(api_error)}'}), 500
        
        if len(messages) >= 3:
            ingest_msg = messages[0]
            retrieval_msg = messages[1] 
            llm_msg = messages[2]
            
            chunks_found = len(ingest_msg.payload.get("chunks", []))
            context_found = len(retrieval_msg.payload.get("retrieved_context", []))
            answer = llm_msg.payload.get("answer", "Sorry, I couldn't generate an answer.")
            source_context = retrieval_msg.payload.get("retrieved_context", [])
            
            # Include similarity information for debugging
            similarities = retrieval_msg.payload.get("similarities", [])
            max_similarity = retrieval_msg.payload.get("max_similarity", 0.0)
            threshold_met = retrieval_msg.payload.get("threshold_met", True)
            
            conversation = session.get('conversation', [])
            conversation.append({
                'user': user_message,
                'assistant': answer,
                'metadata': {
                    'chunks_found': chunks_found,
                    'context_found': context_found,
                    'source_context': source_context[:3],
                    'max_similarity': max_similarity,
                    'threshold_met': threshold_met
                }
            })
            session['conversation'] = conversation
            session.modified = True
            
            return jsonify({
                'success': True,
                'response': answer,
                'metadata': {
                    'chunks_found': chunks_found,
                    'context_found': context_found,
                    'files_processed': len(file_paths),
                    'max_similarity': round(max_similarity, 3) if max_similarity else 0,
                    'threshold_met': threshold_met
                },
                'source_context': source_context[:3]
            })
        else:
            return jsonify({'error': 'Processing incomplete. Please try again.'}), 500
            
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
            'service': 'Agentic RAG Chatbot',
            'has_files': has_files,
            'timestamp': datetime.now().isoformat()
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
