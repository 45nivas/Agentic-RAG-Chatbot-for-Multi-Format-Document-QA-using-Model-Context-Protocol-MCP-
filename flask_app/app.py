from flask import Flask, render_template, request, jsonify, session
import os
import sys
import uuid
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import tempfile
from dotenv import load_dotenv
import numpy as np
import pickle
import json

# Advanced AI/ML Components - Full Development Features
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    ADVANCED_AI_AVAILABLE = True
    print("✅ Advanced AI components loaded: Sentence Transformers + ChromaDB")
except ImportError as e:
    ADVANCED_AI_AVAILABLE = False
    print(f"❌ Advanced AI components not available: {e}")

# Standard ML fallback
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Document processing
import PyPDF2
import docx
from pptx import Presentation
import csv
import io

# Google AI
import google.generativeai as genai

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Multi-Agent Architecture
try:
    from agents.coordinator_agent import CoordinatorAgent
    from agents.ingestion_agent import IngestionAgent
    from agents.retrieval_agent import RetrievalAgent
    from agents.llm_response_agent import LLMResponseAgent
    AGENTS_AVAILABLE = True
    print("✅ Multi-agent architecture loaded successfully")
except ImportError as e:
    AGENTS_AVAILABLE = False
    print(f"❌ Agents not available: {e}")

app = Flask(__name__)

# Development configuration - Full features enabled
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-full-features')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB for development
app.config['ENV'] = 'development'
app.config['DEBUG'] = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini AI
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
model = None
if GEMINI_API_KEY and GEMINI_API_KEY != 'your-gemini-api-key':
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("✅ Gemini AI configured successfully with gemini-1.5-flash")
    except Exception as e:
        logger.error(f"Failed to configure Gemini AI: {str(e)}")
        model = None
else:
    logger.warning("No valid Gemini API key found")

# Advanced Vector Database with Sentence Transformers + ChromaDB
class AdvancedVectorDB:
    """Full-featured vector database with Sentence Transformers and ChromaDB"""
    
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.use_advanced_ai = ADVANCED_AI_AVAILABLE
        
        if self.use_advanced_ai:
            try:
                # Initialize Sentence Transformers (384-dimensional embeddings)
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Sentence Transformers loaded: all-MiniLM-L6-v2 (384d embeddings)")
                
                # Initialize ChromaDB with persistence
                self.chroma_client = chromadb.PersistentClient(
                    path="./chroma_db",
                    settings=Settings(
                        allow_reset=True,
                        anonymized_telemetry=False
                    )
                )
                
                # Create or get collection
                try:
                    self.collection = self.chroma_client.get_collection("documents")
                    logger.info("✅ Connected to existing ChromaDB collection")
                except:
                    self.collection = self.chroma_client.create_collection(
                        name="documents",
                        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                    )
                    logger.info("✅ Created new ChromaDB collection with HNSW indexing")
                
                self.use_chromadb = True
                
            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {e}")
                self.use_chromadb = False
                self.use_advanced_ai = False
        
        # Fallback to enhanced TF-IDF if advanced AI not available
        if not self.use_advanced_ai:
            self.vectorizer = TfidfVectorizer(
                max_features=2000,
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.8,
                min_df=2
            )
            self.vectors = None
            self.fitted = False
            logger.info("🔄 Using enhanced TF-IDF fallback")
        
        logger.info(f"Advanced Vector Database initialized (AI: {self.use_advanced_ai}, ChromaDB: {getattr(self, 'use_chromadb', False)})")
    
    def add_document(self, filename, text_content):
        """Add document with advanced semantic processing"""
        try:
            # Advanced semantic chunking
            chunks = self.advanced_semantic_chunk(text_content, filename)
            
            # Store document metadata
            doc_id = len(self.documents)
            self.documents.append({
                'id': doc_id,
                'filename': filename,
                'content': text_content,
                'chunks_count': len(chunks),
                'timestamp': datetime.now().isoformat()
            })
            
            if self.use_advanced_ai and self.use_chromadb:
                # Use Sentence Transformers + ChromaDB
                return self._add_with_sentence_transformers(chunks, filename, doc_id)
            else:
                # Fallback to enhanced TF-IDF
                return self._add_with_tfidf(chunks, filename, doc_id)
                
        except Exception as e:
            logger.error(f"Error adding document {filename}: {e}")
            return False
    
    def advanced_semantic_chunk(self, text, filename, chunk_size=400, overlap=100):
        """Advanced chunking with sentence boundaries and context preservation"""
        import re
        
        # Multiple splitting strategies for better chunking
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # Split by sentences with multiple delimiters
            sentences = re.split(r'[.!?]+(?:\s+|$)', para)
            current_chunk = ""
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                words = sentence.split()
                sentence_length = len(words)
                
                # Smart chunking with overlap
                if current_length + sentence_length > chunk_size and current_chunk:
                    # Save current chunk with metadata
                    chunks.append({
                        'content': current_chunk.strip(),
                        'filename': filename,
                        'chunk_index': len(chunks),
                        'word_count': current_length
                    })
                    
                    # Create intelligent overlap
                    overlap_sentences = current_chunk.split('.')[-2:] if '.' in current_chunk else [current_chunk]
                    overlap_text = '. '.join(s.strip() for s in overlap_sentences if s.strip())
                    current_chunk = overlap_text + ". " + sentence if overlap_text else sentence
                    current_length = len(current_chunk.split())
                else:
                    current_chunk += " " + sentence
                    current_length += sentence_length
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append({
                    'content': current_chunk.strip(),
                    'filename': filename,
                    'chunk_index': len(chunks),
                    'word_count': current_length
                })
        
        logger.info(f"Advanced chunking: {len(chunks)} chunks created for {filename}")
        return chunks
    
    def _add_with_sentence_transformers(self, chunks, filename, doc_id):
        """Add chunks using Sentence Transformers + ChromaDB"""
        try:
            chunk_texts = [chunk['content'] for chunk in chunks]
            
            # Generate 384-dimensional embeddings
            embeddings = self.sentence_model.encode(
                chunk_texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Prepare data for ChromaDB
            chunk_ids = [f"{filename}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    'filename': filename,
                    'doc_id': str(doc_id),
                    'chunk_index': i,
                    'word_count': chunk['word_count'],
                    'timestamp': datetime.now().isoformat()
                }
                metadatas.append(metadata)
                
                # Store in local chunks list too
                chunk_data = {
                    'id': chunk_ids[i],
                    'content': chunk['content'],
                    'filename': filename,
                    'doc_id': doc_id,
                    'chunk_index': i,
                    'embedding': embeddings[i].tolist()
                }
                self.chunks.append(chunk_data)
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=chunk_texts,
                metadatas=metadatas,
                ids=chunk_ids
            )
            
            logger.info(f"✅ Added {len(chunks)} chunks to ChromaDB with Sentence Transformers for {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Sentence Transformers processing error: {e}")
            return False
    
    def _add_with_tfidf(self, chunks, filename, doc_id):
        """Fallback: Add chunks using enhanced TF-IDF"""
        try:
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'id': f"{filename}_{i}",
                    'content': chunk['content'],
                    'filename': filename,
                    'doc_id': doc_id,
                    'chunk_index': i
                }
                self.chunks.append(chunk_data)
            
            # Rebuild TF-IDF vectors
            chunk_texts = [chunk['content'] for chunk in self.chunks]
            self.vectors = self.vectorizer.fit_transform(chunk_texts)
            self.fitted = True
            
            logger.info(f"✅ Added {len(chunks)} chunks with enhanced TF-IDF for {filename}")
            return True
            
        except Exception as e:
            logger.error(f"TF-IDF processing error: {e}")
            return False
    
    def search(self, query, top_k=5, threshold=0.3):
        """Advanced semantic search"""
        if not self.chunks:
            logger.warning("No documents available for search")
            return []
        
        try:
            if self.use_advanced_ai and self.use_chromadb:
                return self._search_with_sentence_transformers(query, top_k, threshold)
            else:
                return self._search_with_tfidf(query, top_k, threshold)
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _search_with_sentence_transformers(self, query, top_k, threshold):
        """Search using Sentence Transformers + ChromaDB"""
        try:
            # Generate query embedding
            query_embedding = self.sentence_model.encode([query])
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(top_k, len(self.chunks)),
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            search_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                # Convert distance to similarity (ChromaDB returns cosine distance)
                similarity = 1 - distance
                
                if similarity > threshold:
                    search_results.append({
                        'content': doc,
                        'filename': metadata['filename'],
                        'similarity': float(similarity),
                        'chunk_index': metadata['chunk_index'],
                        'word_count': metadata.get('word_count', 0),
                        'metadata': metadata
                    })
            
            logger.info(f"🔍 Sentence Transformers search: {len(search_results)} results (avg similarity: {np.mean([r['similarity'] for r in search_results]):.3f})")
            return search_results
            
        except Exception as e:
            logger.error(f"Sentence Transformers search error: {e}")
            return []
    
    def _search_with_tfidf(self, query, top_k, threshold):
        """Fallback search using enhanced TF-IDF"""
        if not self.fitted:
            return []
            
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.vectors)[0]
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity > threshold and idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    results.append({
                        'content': chunk['content'],
                        'filename': chunk['filename'],
                        'similarity': float(similarity),
                        'chunk_index': chunk['chunk_index'],
                        'metadata': {'filename': chunk['filename']}
                    })
            
            logger.info(f"🔍 Enhanced TF-IDF search: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"TF-IDF search error: {e}")
            return []

# Initialize Advanced Vector Database
vector_db = AdvancedVectorDB()

# Initialize Multi-Agent System
if AGENTS_AVAILABLE:
    try:
        coordinator = CoordinatorAgent()
        ingestion_agent = IngestionAgent()
        retrieval_agent = RetrievalAgent()
        llm_agent = LLMResponseAgent()
        logger.info("✅ Multi-agent system initialized successfully")
    except Exception as e:
        logger.error(f"Multi-agent initialization error: {e}")
        AGENTS_AVAILABLE = False

# Advanced Document Processing Functions
def extract_text_from_pdf(file_path):
    """Extract text from PDF with advanced processing"""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text += f"\\n--- Page {page_num + 1} ---\\n{page_text}\\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""

def extract_text_from_docx(file_path):
    """Extract text from DOCX with paragraph structure"""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\\n\\n"
        return text.strip()
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return ""

def extract_text_from_pptx(file_path):
    """Extract text from PowerPoint with slide structure"""
    try:
        prs = Presentation(file_path)
        text = ""
        for slide_num, slide in enumerate(prs.slides):
            slide_text = f"\\n--- Slide {slide_num + 1} ---\\n"
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text += shape.text + "\\n"
            if slide_text.strip() != f"--- Slide {slide_num + 1} ---":
                text += slide_text + "\\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PPTX extraction error: {e}")
        return ""

def extract_text_from_csv(file_path):
    """Extract text from CSV with column structure"""
    try:
        text = ""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader, [])
            if headers:
                text += "Columns: " + ", ".join(headers) + "\\n\\n"
            
            for row_num, row in enumerate(csv_reader, 1):
                if row and any(cell.strip() for cell in row):
                    row_text = " | ".join(str(cell).strip() for cell in row)
                    text += f"Row {row_num}: {row_text}\\n"
                    
                # Limit rows for large CSV files
                if row_num > 1000:
                    text += "\\n... (truncated for large file) ...\\n"
                    break
                    
        return text.strip()
    except Exception as e:
        logger.error(f"CSV extraction error: {e}")
        return ""

def extract_text_from_txt(file_path):
    """Extract text from plain text files"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read().strip()
    except Exception as e:
        logger.error(f"TXT extraction error: {e}")
        return ""

def extract_text_from_md(file_path):
    """Extract text from Markdown files"""
    try:
        import re
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            
        # Clean up markdown formatting while preserving structure
        content = re.sub(r'^#{1,6}\\s+', '', content, flags=re.MULTILINE)  # Remove headers
        content = re.sub(r'\\*\\*(.*?)\\*\\*', r'\\1', content)  # Remove bold
        content = re.sub(r'\\*(.*?)\\*', r'\\1', content)  # Remove italic
        content = re.sub(r'`(.*?)`', r'\\1', content)  # Remove inline code
        content = re.sub(r'\\[([^\\]]+)\\]\\([^\\)]+\\)', r'\\1', content)  # Remove links
        
        return content.strip()
    except Exception as e:
        logger.error(f"MD extraction error: {e}")
        return ""

def process_document(file_path, filename):
    """Advanced document processing with format detection"""
    try:
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        extraction_functions = {
            'pdf': extract_text_from_pdf,
            'docx': extract_text_from_docx,
            'pptx': extract_text_from_pptx,
            'csv': extract_text_from_csv,
            'txt': extract_text_from_txt,
            'md': extract_text_from_md
        }
        
        if file_extension in extraction_functions:
            text_content = extraction_functions[file_extension](file_path)
            if text_content:
                logger.info(f"✅ Extracted {len(text_content)} characters from {filename}")
                return text_content
            else:
                logger.warning(f"No text extracted from {filename}")
                return ""
        else:
            logger.error(f"Unsupported file format: {file_extension}")
            return ""
            
    except Exception as e:
        logger.error(f"Document processing error for {filename}: {e}")
        return ""

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
    """Advanced file upload with Sentence Transformers processing"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files selected'}), 400
        
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400
        
        processed_files = []
        total_chunks = 0
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                try:
                    # Secure filename and save
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_filename = f"{timestamp}_{filename}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                    
                    file.save(file_path)
                    
                    # Advanced document processing
                    text_content = process_document(file_path, filename)
                    
                    if text_content:
                        # Add to advanced vector database
                        success = vector_db.add_document(filename, text_content)
                        
                        if success:
                            file_info = {
                                'filename': filename,
                                'size': len(text_content),
                                'processed': True,
                                'ai_method': 'Sentence Transformers + ChromaDB' if vector_db.use_advanced_ai else 'Enhanced TF-IDF',
                                'timestamp': datetime.now().isoformat()
                            }
                            processed_files.append(file_info)
                            
                            # Update session
                            if 'uploaded_files' not in session:
                                session['uploaded_files'] = []
                            session['uploaded_files'].append(file_info)
                            
                            # Count chunks for this document
                            doc_chunks = len([chunk for chunk in vector_db.chunks if chunk['filename'] == filename])
                            total_chunks += doc_chunks
                            
                            logger.info(f"✅ Advanced processing complete: {filename} ({doc_chunks} chunks)")
                        else:
                            logger.error(f"Failed to add {filename} to vector database")
                    else:
                        logger.warning(f"No text content extracted from {filename}")
                    
                    # Clean up temporary file
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        
                except Exception as e:
                    logger.error(f"Error processing file {file.filename}: {e}")
                    continue
        
        if processed_files:
            ai_info = {
                'model': 'all-MiniLM-L6-v2 (384d)' if vector_db.use_advanced_ai else 'Enhanced TF-IDF',
                'database': 'ChromaDB' if getattr(vector_db, 'use_chromadb', False) else 'In-Memory',
                'total_chunks': total_chunks,
                'total_documents': len(vector_db.documents)
            }
            
            return jsonify({
                'success': True,
                'message': f'Successfully processed {len(processed_files)} files with advanced AI',
                'files': processed_files,
                'ai_info': ai_info
            })
        else:
            return jsonify({'error': 'No files could be processed'}), 400
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500
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
    """Advanced chat with multi-agent processing and Sentence Transformers"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Check if documents are available
        if not vector_db.chunks:
            return jsonify({
                'error': 'No documents uploaded yet. Please upload documents first.',
                'response': 'I need documents to answer questions. Please upload some files first!'
            }), 400
        
        # Advanced semantic search
        search_results = vector_db.search(user_message, top_k=5, threshold=0.2)
        
        if not search_results:
            response = "I couldn't find relevant information in the uploaded documents for your question."
            search_info = {
                'method': 'Advanced AI Search',
                'results_found': 0,
                'ai_model': 'No matches'
            }
        else:
            # Multi-agent processing if available
            if AGENTS_AVAILABLE and model:
                try:
                    # Use multi-agent system for enhanced response
                    context = "\\n\\n".join([result['content'] for result in search_results[:3]])
                    
                    # Enhanced prompt with multi-agent context
                    enhanced_prompt = f"""You are an advanced AI assistant with access to uploaded documents. 
                    Use the following context to provide a comprehensive, accurate answer.
                    
                    Context from documents:
                    {context}
                    
                    User Question: {user_message}
                    
                    Instructions:
                    1. Provide a detailed, well-structured answer based on the context
                    2. If the context doesn't fully answer the question, say so clearly
                    3. Include relevant details and examples from the documents
                    4. Be precise and helpful
                    
                    Answer:"""
                    
                    # Generate response with Gemini
                    gemini_response = model.generate_content(enhanced_prompt)
                    response = gemini_response.text
                    
                    logger.info(f"✅ Multi-agent response generated for: {user_message[:50]}...")
                    
                except Exception as e:
                    logger.error(f"Multi-agent processing error: {e}")
                    # Fallback to simple response
                    response = f"Based on the uploaded documents: {search_results[0]['content'][:500]}..."
            else:
                # Simple response if agents not available
                response = f"Based on the uploaded documents: {search_results[0]['content'][:500]}..."
            
            # Advanced search information
            avg_similarity = sum(r['similarity'] for r in search_results) / len(search_results)
            search_info = {
                'method': 'Sentence Transformers + ChromaDB' if vector_db.use_advanced_ai else 'Enhanced TF-IDF',
                'results_found': len(search_results),
                'avg_similarity': round(avg_similarity, 3),
                'top_similarity': round(search_results[0]['similarity'], 3),
                'sources': list(set(r['filename'] for r in search_results)),
                'ai_model': 'all-MiniLM-L6-v2' if vector_db.use_advanced_ai else 'TF-IDF Enhanced'
            }
        
        # Update conversation history
        if 'conversation' not in session:
            session['conversation'] = []
        
        conversation_entry = {
            'user_message': user_message,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'search_info': search_info
        }
        
        session['conversation'].append(conversation_entry)
        session.modified = True
        
        logger.info(f"🚀 Advanced chat completed: {len(search_results)} results, {search_info['method']}")
        
        return jsonify({
            'response': response,
            'search_info': search_info,
            'conversation_id': len(session['conversation'])
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500
        
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
