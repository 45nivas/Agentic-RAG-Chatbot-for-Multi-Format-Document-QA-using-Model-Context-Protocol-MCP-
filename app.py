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

# Production-Ready Components - Lightweight for Render Deployment
# Using optimized TF-IDF for reliable cloud deployment within memory limits 

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json

# Multi-Agent Architecture
try:
    from agents.coordinator_agent import CoordinatorAgent
    from agents.ingestion_agent import IngestionAgent
    from agents.retrieval_agent import RetrievalAgent
    from agents.llm_response_agent import LLMResponseAgent
    from agents.document_utils import parse_document
    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENTS_AVAILABLE = False
    parse_document = None

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
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
model = None
if GEMINI_API_KEY and GEMINI_API_KEY != 'your-gemini-api-key':
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Try multiple models in case quota varies
        models_to_try = ['models/gemini-2.0-flash-exp', 'models/gemini-2.0-flash', 'models/gemini-2.5-flash']
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                # Quick test
                test_response = model.generate_content("Hi")
                logger.info(f"✅ Gemini AI configured successfully with {model_name}")
                break
            except Exception as model_error:
                logger.warning(f"⚠️ Failed with {model_name}: {str(model_error)[:100]}")
                continue
        
        if model is None:
            logger.error("❌ All Gemini models failed - likely quota exceeded")
            
    except Exception as e:
        logger.error(f"❌ Failed to configure Gemini AI: {str(e)}")
        model = None
else:
    logger.warning("⚠️ No valid Gemini API key found in environment variables")

# Initialize Multi-Agent System with ChromaDB
if AGENTS_AVAILABLE:
    try:
        from agents.embedding_utils import EmbeddingStore
        
        coordinator = CoordinatorAgent()
        ingestion_agent = IngestionAgent()
        retrieval_agent = RetrievalAgent()
        llm_agent = LLMResponseAgent()
        
        # Initialize shared vector store with ChromaDB
        embedding_store = EmbeddingStore(persist_path="./chroma_render")
        retrieval_agent.vector_store = embedding_store
        
        logger.info("✅ Multi-agent system with ChromaDB initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize agents: {str(e)}")
        AGENTS_AVAILABLE = False
        embedding_store = None
else:
    coordinator = None
    ingestion_agent = None
    retrieval_agent = None
    llm_agent = None
    embedding_store = None

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'pptx', 'csv', 'txt', 'md'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def store_document_in_vector_db(filename, text_content):
    """Store document using RetrievalAgent + ChromaDB"""
    if not embedding_store or not ingestion_agent:
        return False
    
    try:
        # Use IngestionAgent to parse and chunk
        chunks = parse_document(filename) if parse_document else [text_content]
        
        # Store in RetrievalAgent's ChromaDB
        embedding_store.add_chunks(chunks)
        logger.info(f"✅ Added {len(chunks)} chunks to ChromaDB for {filename}")
        return True
    except Exception as e:
        logger.error(f"Storage failed: {str(e)}")
        return False

def retrieve_relevant_context(query, top_k=5):
    """Retrieve using RetrievalAgent + ChromaDB"""
    if not retrieval_agent:
        return []
    
    try:
        # Use RetrievalAgent to search
        retrieval_msg = retrieval_agent.embed_and_retrieve([], query, top_k=top_k)
        contexts = retrieval_msg.payload.get('retrieved_context', [])
        similarities = retrieval_msg.payload.get('similarities', [])
        
        # Format for app
        formatted = []
        for i, ctx in enumerate(contexts):
            formatted.append({
                'content': ctx,
                'filename': 'document',
                'similarity': similarities[i] if i < len(similarities) else 0.5
            })
        
        logger.info(f"✅ RetrievalAgent retrieved {len(formatted)} contexts")
        return formatted
    except Exception as e:
        logger.error(f"Retrieval failed: {str(e)}")
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
                
                # Use agents to parse document
                if parse_document:
                    chunks = parse_document(filepath)
                    text_content = ' '.join(chunks) if chunks else ''
                else:
                    text_content = ''
                
                # Store in vector database for professional RAG
                vector_stored = store_document_in_vector_db(filename, text_content)
                
                uploaded_files.append({
                    'filename': filename,
                    'filepath': filepath,
                    'content': text_content[:2000],  # Store first 2000 chars for session
                    'vector_stored': vector_stored,
                    'chunks_count': len(chunks) if chunks else 0
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
            # Fallback to stored content if vector DB fails
            context_text = ""
            for file_info in uploaded_files:
                content = file_info.get('content', '')
                filename = file_info.get('filename')
                if content:
                    context_text += f"\n\n=== Content from {filename} ===\n{content[:3000]}\n"
            
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
                error_msg = str(e)
                logger.error(f"Gemini API error: {error_msg}")
                
                # Check if quota exceeded
                if "429" in error_msg or "quota" in error_msg.lower():
                    ai_response = "🚫 Daily AI quota exceeded. Here's the most relevant content from your documents:\n\n"
                    if relevant_contexts:
                        ai_response += relevant_contexts[0]['content'][:800]
                    else:
                        ai_response += "Please try again tomorrow or upgrade your API plan."
                else:
                    # Other API errors - provide fallback content
                    if relevant_contexts:
                        best_context = relevant_contexts[0]['content'][:500]
                        ai_response = f"Based on your documents: {best_context}..." if len(best_context) == 500 else f"Based on your documents: {best_context}"
                    else:
                        ai_response = "I found your documents but couldn't process your question. Please try rephrasing it."
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
        embedding_mode = 'tfidf' if embedding_store and embedding_store.use_tfidf else 'sentence_transformers'
        
        return jsonify({
            'status': 'healthy',
            'service': 'Agentic RAG with ChromaDB',
            'has_files': has_files,
            'ai_enabled': bool(model),
            'vector_db_enabled': bool(embedding_store),
            'embedding_mode': embedding_mode,
            'multi_agent_enabled': AGENTS_AVAILABLE,
            'features': [
                'chromadb',
                'multi_agent_coordination',
                'mcp_protocol',
                'voice_input',
                '6_file_formats',
                embedding_mode + '_embeddings'
            ],
            'timestamp': datetime.now().isoformat(),
            'version': '4.0.0'
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

if __name__ == '__main__':
    # Production-ready server configuration
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = app.config['DEBUG']
    
    logger.info(f"Starting RAG Chatbot server on {host}:{port}")
    app.run(debug=debug, host=host, port=port)
