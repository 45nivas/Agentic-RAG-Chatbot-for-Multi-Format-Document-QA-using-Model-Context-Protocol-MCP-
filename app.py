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

# Modern AI/ML Components - Sentence Transformers & Vector DB
from sentence_transformers import SentenceTransformer
import chromadb
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json

# Multi-Agent Architecture
try:
    from agents.coordinator_agent import CoordinatorAgent
    from agents.ingestion_agent import IngestionAgent
    from agents.retrieval_agent import RetrievalAgent
    from agents.llm_response_agent import LLMResponseAgent
    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENTS_AVAILABLE = False

load_dotenv()

app = Flask(__name__)

# Production-ready configuration - Updated for deployment
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB for production
app.config['ENV'] = os.environ.get('FLASK_ENV', 'production')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log agent availability
if AGENTS_AVAILABLE:
    logger.info("Multi-agent architecture loaded successfully")
else:
    logger.warning("Agents not available, using direct implementation")

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

# Modern Professional Vector Database with Sentence Transformers
class ModernVectorDB:
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.metadata = []
        
        # Initialize ChromaDB for persistent vector storage
        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.warning(f"ChromaDB init failed, using in-memory: {e}")
            self.chroma_client = None
            self.collection = None
        
        # Initialize modern embedding model - Sentence Transformers
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer model loaded: all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            self.embedding_model = None
        
        # Fallback to in-memory vectors if needed
        self.vectors = None
        
    def add_document(self, filename, text_content):
        """Add document with modern semantic chunking and embeddings"""
        try:
            if not self.embedding_model:
                logger.error("No embedding model available")
                return False
                
            # Advanced semantic chunking
            chunks = self.semantic_chunk_text(text_content)
            
            # Add to storage
            doc_id = len(self.documents)
            self.documents.append({
                'id': doc_id,
                'filename': filename,
                'content': text_content,
                'chunks_count': len(chunks)
            })
            
            # Generate embeddings using Sentence Transformers
            embeddings = self.embedding_model.encode(chunks).tolist()
            
            # Add chunks with metadata
            for i, chunk in enumerate(chunks):
                self.chunks.append(chunk)
                chunk_metadata = {
                    'doc_id': doc_id,
                    'chunk_id': i,
                    'filename': filename,
                    'chunk_type': 'semantic'
                }
                self.metadata.append(chunk_metadata)
                
                # Store in ChromaDB if available
                if self.collection:
                    try:
                        self.collection.add(
                            embeddings=[embeddings[i]],
                            documents=[chunk],
                            metadatas=[chunk_metadata],
                            ids=[f"{filename}_{doc_id}_{i}"]
                        )
                    except Exception as e:
                        logger.warning(f"ChromaDB storage failed: {e}")
            
            # Store embeddings in memory as backup
            if self.vectors is None:
                self.vectors = np.array(embeddings)
            else:
                self.vectors = np.vstack([self.vectors, embeddings])
            
            logger.info(f"Added {len(chunks)} semantic chunks from {filename} with embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document to modern vector DB: {str(e)}")
            return False
    
    def semantic_chunk_text(self, text, chunk_size=300, overlap=50):
        """Advanced semantic text chunking optimized for embeddings"""
        # Split by sentences first for better semantic boundaries
        sentences = text.replace('\n', ' ').split('. ')
        
        chunks = []
        current_chunk = ""
        word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed chunk_size, save current chunk
            if word_count + sentence_words > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_words = current_chunk.split()[-overlap:]
                current_chunk = ' '.join(overlap_words) + ' ' + sentence
                word_count = len(overlap_words) + sentence_words
            else:
                current_chunk += ' ' + sentence
                word_count += sentence_words
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def search(self, query, top_k=5):
        """Modern semantic search using sentence transformers"""
        if not self.chunks or not self.embedding_model:
            return []
        
        try:
            # First try ChromaDB search
            if self.collection:
                query_embedding = self.embedding_model.encode([query]).tolist()
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(top_k, len(self.chunks))
                )
                
                search_results = []
                if results['documents'] and results['documents'][0]:
                    for i, doc in enumerate(results['documents'][0]):
                        distance = results['distances'][0][i] if results['distances'] else 0
                        metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                        
                        search_results.append({
                            'content': doc,
                            'similarity': 1 - distance,  # Convert distance to similarity
                            'metadata': metadata
                        })
                
                logger.info(f"ChromaDB search returned {len(search_results)} results")
                return search_results
            
            # Fallback to in-memory search
            query_embedding = self.embedding_model.encode([query])
            
            # Calculate similarities using embeddings
            similarities = cosine_similarity(query_embedding, self.vectors)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.2:  # Higher threshold for semantic similarity
                    results.append({
                        'content': self.chunks[idx],
                        'similarity': similarities[idx],
                        'metadata': self.metadata[idx]
                    })
            
            logger.info(f"Semantic search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []

# Initialize Modern Professional Components
try:
    vector_db = ModernVectorDB()
    logger.info("Modern Vector Database with SentenceTransformers initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Modern Vector DB: {str(e)}")
    vector_db = None

# Initialize Multi-Agent System
if AGENTS_AVAILABLE:
    try:
        coordinator = CoordinatorAgent()
        ingestion_agent = IngestionAgent()
        retrieval_agent = RetrievalAgent()
        llm_agent = LLMResponseAgent()
        logger.info("Multi-agent system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agents: {str(e)}")
        AGENTS_AVAILABLE = False
else:
    coordinator = None
    ingestion_agent = None
    retrieval_agent = None
    llm_agent = None

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
        
        # Modern Multi-Agent RAG Processing
        if AGENTS_AVAILABLE and coordinator:
            try:
                # Use coordinated multi-agent approach
                response = coordinator.process_query(
                    query=user_message,
                    documents=uploaded_files,
                    session_id=session.get('session_id', str(uuid.uuid4()))
                )
                
                conversation = session.get('conversation', [])
                conversation.append({
                    'user': user_message,
                    'assistant': response.content,
                    'metadata': {
                        'agent_type': 'multi_agent',
                        'similarity_scores': response.metadata.get('similarities', []),
                        'sources': response.metadata.get('sources', []),
                        'processing_time': response.metadata.get('time', 0),
                        'timestamp': datetime.now().isoformat()
                    }
                })
                session['conversation'] = conversation
                
                return jsonify({
                    'response': response.content,
                    'metadata': response.metadata,
                    'agent_type': 'multi_agent_coordinator'
                })
                
            except Exception as e:
                logger.warning(f"Multi-agent processing failed, falling back: {e}")
        
        # Fallback: Modern Vector Search with Sentence Transformers
        relevant_contexts = retrieve_relevant_context(user_message, top_k=5)
        
        if relevant_contexts:
            # Use semantic vector search results
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
        
        # Check modern capabilities
        embedding_available = vector_db and vector_db.embedding_model is not None
        chromadb_available = vector_db and vector_db.collection is not None
        
        return jsonify({
            'status': 'healthy',
            'service': 'Modern Agentic RAG System with Sentence Transformers',
            'has_files': has_files,
            'ai_enabled': bool(model),
            'vector_db_enabled': bool(vector_db),
            'sentence_transformers_enabled': embedding_available,
            'chromadb_enabled': chromadb_available,
            'multi_agent_enabled': AGENTS_AVAILABLE,
            'rag_mode': 'semantic_embeddings' if embedding_available else 'fallback',
            'features': [
                'semantic_search', 
                'sentence_transformers', 
                'chromadb_storage',
                'multi_agent_coordination',
                'advanced_chunking',
                'voice_input'
            ],
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
