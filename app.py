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
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json

load_dotenv()

app = Flask(__name__)

# Production-ready configuration - Updated for deployment
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB for production
app.config['ENV'] = os.environ.get('FLASK_ENV', 'production')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Professional RAG System - Lightweight Implementation
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyCyuSLpjW7e7z4tmWcqHzg5dNQTmlzEqGc')
model = None
if GEMINI_API_KEY and GEMINI_API_KEY != 'your-gemini-api-key':
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Gemini AI configured successfully with gemini-1.5-flash")
    except Exception as e:
        logger.error(f"Failed to configure Gemini AI: {str(e)}")
        model = None
else:
    logger.warning("No valid Gemini API key found")

# Professional Vector Storage System (In-Memory)
class ProfessionalVectorDB:
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.vectors = None
        self.metadata = []
        
    def add_document(self, filename, text_content):
        """Add document with chunking and vectorization"""
        try:
            # Chunk the document
            chunks = self.chunk_text(text_content)
            
            # Add to storage
            doc_id = len(self.documents)
            self.documents.append({
                'id': doc_id,
                'filename': filename,
                'content': text_content,
                'chunks_count': len(chunks)
            })
            
            # Add chunks with metadata
            for i, chunk in enumerate(chunks):
                self.chunks.append(chunk)
                self.metadata.append({
                    'doc_id': doc_id,
                    'chunk_id': i,
                    'filename': filename
                })
            
            # Rebuild vectors
            self._rebuild_vectors()
            logger.info(f"Added {len(chunks)} chunks from {filename} to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document to vector DB: {str(e)}")
            return False
    
    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Professional text chunking with overlap"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks
    
    def _rebuild_vectors(self):
        """Rebuild TF-IDF vectors for all chunks"""
        if self.chunks:
            self.vectors = self.vectorizer.fit_transform(self.chunks)
    
    def search(self, query, top_k=5):
        """Professional similarity search"""
        if not self.chunks or self.vectors is None:
            return []
        
        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.vectors)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    results.append({
                        'content': self.chunks[idx],
                        'similarity': similarities[idx],
                        'metadata': self.metadata[idx]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

# Initialize Professional Components
try:
    vector_db = ProfessionalVectorDB()
    logger.info("Professional Vector Database initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Professional Vector DB: {str(e)}")
    vector_db = None

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

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks for better context"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

def store_document_in_vector_db(filename, text_content):
    """Store document in professional vector database"""
    if not vector_db:
        return False
    
    try:
        return vector_db.add_document(filename, text_content)
        
    except Exception as e:
        logger.error(f"Failed to store document in vector DB: {str(e)}")
        return False

def retrieve_relevant_context(query, top_k=5):
    """Retrieve most relevant document chunks using professional vector similarity"""
    if not vector_db:
        return []
    
    try:
        # Search for similar chunks
        results = vector_db.search(query, top_k=top_k)
        
        # Format results
        contexts = []
        for result in results:
            contexts.append({
                'content': result['content'],
                'filename': result['metadata']['filename'],
                'similarity': result['similarity']
            })
        
        logger.info(f"Retrieved {len(contexts)} relevant contexts for query")
        return contexts
        
    except Exception as e:
        logger.error(f"Failed to retrieve context: {str(e)}")
        return []

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
                
                # Store in vector database for professional RAG
                vector_stored = store_document_in_vector_db(filename, text_content)
                
                uploaded_files.append({
                    'filename': filename,
                    'filepath': filepath,
                    'content': text_content[:2000],  # Store first 2000 chars for session
                    'vector_stored': vector_stored,
                    'chunks_count': len(chunk_text(text_content))
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
        
        # Professional RAG: Retrieve relevant context using vector similarity
        relevant_contexts = retrieve_relevant_context(user_message, top_k=5)
        
        if relevant_contexts:
            # Use vector search results (Professional RAG)
            context_text = "\n\n".join([f"From {ctx['filename']}: {ctx['content']}" 
                                       for ctx in relevant_contexts[:3]])
            
            similarity_scores = [ctx['similarity'] for ctx in relevant_contexts]
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            
            logger.info(f"Using vector search with {len(relevant_contexts)} chunks, avg similarity: {avg_similarity:.3f}")
            
        else:
            # Fallback to simple text search if vector DB fails
            context_text = ""
            for file_info in uploaded_files:
                filepath = file_info.get('filepath')
                filename = file_info.get('filename')
                if filepath and os.path.exists(filepath):
                    full_content = extract_text_from_file(filepath, filename)
                    context_text += f"\n\n=== Content from {filename} ===\n{full_content[:3000]}\n"
            
            avg_similarity = 0.5  # Default similarity for fallback
            logger.info("Using fallback text search (vector DB unavailable)")

        # Generate AI response using professional RAG context
        if model and context_text:
            try:
                prompt = f"""You are a professional document analysis assistant. Answer the user's question based on the provided context.

RELEVANT CONTEXT:
{context_text}

USER QUESTION: {user_message}

Provide a detailed, accurate answer based on the context. If the information isn't in the context, say so clearly."""

                generation_config = {
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 1024,
                }
                
                response = model.generate_content(prompt, generation_config=generation_config)
                ai_response = response.text.strip()
                
                if not ai_response:
                    ai_response = f"I processed your documents using professional RAG (vector similarity: {avg_similarity:.1%}), but couldn't generate a response. Please try rephrasing your question."
                    
                logger.info("Successfully generated professional RAG response")
                
            except Exception as e:
                logger.error(f"Gemini API error: {str(e)}")
                ai_response = f"Professional RAG system processed your query with {len(relevant_contexts)} relevant chunks (similarity: {avg_similarity:.1%}), but encountered an API issue. Please try again."
        elif not model:
            ai_response = f"I can see you've uploaded {len(uploaded_files)} file(s) including '{uploaded_files[0]['filename']}'. The AI service is currently being configured. Please try again in a moment."
        else:
            ai_response = f"I can see you've uploaded {len(uploaded_files)} file(s), but I need document content to analyze. Please make sure your files contain readable text."
        
        conversation = session.get('conversation', [])
        conversation.append({
            'user': user_message,
            'assistant': ai_response,
            'metadata': {
                'files_processed': len(uploaded_files),
                'has_ai': bool(model),
                'vector_search_used': bool(relevant_contexts),
                'similarity_score': avg_similarity,
                'chunks_retrieved': len(relevant_contexts),
                'rag_mode': 'professional_vector' if relevant_contexts else 'fallback_text',
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
                'has_ai': bool(model),
                'vector_search_used': bool(relevant_contexts),
                'similarity_score': avg_similarity,
                'chunks_retrieved': len(relevant_contexts),
                'rag_mode': 'professional_vector' if relevant_contexts else 'fallback_text',
                'timestamp': datetime.now().isoformat()
            },
            'source_context': [ctx['content'][:200] + '...' for ctx in relevant_contexts[:3]] if relevant_contexts else []
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
            'service': 'Professional Agentic RAG System',
            'has_files': has_files,
            'ai_enabled': bool(model),
            'vector_db_enabled': bool(vector_db),
            'professional_rag_enabled': bool(vector_db),
            'rag_mode': 'professional_tfidf',
            'features': ['vector_search', 'similarity_scoring', 'text_chunking', 'voice_input'],
            'timestamp': datetime.now().isoformat(),
            'version': '3.0.0'
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
