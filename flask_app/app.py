from flask import Flask, render_template, request, jsonify, session
import os
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Advanced AI Stack - Sentence Transformers + ChromaDB (Professional Setup)
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ Sentence Transformers + ChromaDB - Full Advanced AI Stack!")
except ImportError as e:
    # Fallback to ChromaDB only
    import chromadb
    from chromadb.config import Settings
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(f"⚠️ Fallback to ChromaDB embeddings: {e}")
    print("✅ ChromaDB initialized with built-in embeddings")

# Fallback imports REMOVED - Only modern AI
import google.generativeai as genai
import PyPDF2
import docx
import json

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-advanced')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['DEBUG'] = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("✅ Gemini AI configured")
else:
    logger.warning("⚠️ No GEMINI_API_KEY found - responses will be limited")

class AdvancedRAG:
    """Professional RAG with Sentence Transformers + ChromaDB Vector Storage"""
    
    def __init__(self):
        # Initialize advanced Sentence Transformers model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dim = 384  # MiniLM-L6-v2 output dimensions
                logger.info("✅ Sentence Transformers loaded: all-MiniLM-L6-v2 (384D)")
                self.use_sentence_transformers = True
            except Exception as e:
                logger.error(f"Failed to load Sentence Transformers: {e}")
                self.sentence_model = None
                self.embedding_dim = 768  # ChromaDB default
                self.use_sentence_transformers = False
                logger.info("✅ Fallback to ChromaDB built-in embeddings")
        else:
            self.sentence_model = None
            self.embedding_dim = 768  # ChromaDB default
            self.use_sentence_transformers = False
            logger.info("✅ Using ChromaDB built-in embeddings")
        
        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_advanced",
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection("documents")
            logger.info("✅ Connected to existing ChromaDB collection")
        except:
            self.collection = self.chroma_client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}  # Cosine similarity
            )
            logger.info("✅ Created new ChromaDB collection with HNSW indexing")
        
        self.chunks = []  # Keep track of chunks
        logger.info("🚀 Advanced RAG initialized - ONLY modern AI!")
    
    def smart_chunk(self, text, filename, chunk_size=400, overlap=50):
        """Smart chunking with sentence boundaries"""
        import re
        
        # Split by sentences
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            words = sentence.split()
            
            if len(current_chunk) + len(sentence) > chunk_size * 4 and current_chunk:
                chunks.append(current_chunk.strip())
                # Add overlap
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def add_document(self, filename, content):
        """Add document with Sentence Transformers or ChromaDB embeddings"""
        try:
            # Smart chunking
            chunks = self.smart_chunk(content, filename)
            
            # Prepare data for ChromaDB
            chunk_ids = [f"{filename}_{i}" for i in range(len(chunks))]
            metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]
            
            if self.use_sentence_transformers and self.sentence_model:
                # Advanced: Use Sentence Transformers for high-quality embeddings
                embeddings = self.sentence_model.encode(chunks)
                
                # Store in ChromaDB with custom embeddings
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=chunks,
                    metadatas=metadatas,
                    ids=chunk_ids
                )
                
                # Store locally too
                for i, chunk in enumerate(chunks):
                    self.chunks.append({
                        "id": chunk_ids[i],
                        "content": chunk,
                        "filename": filename,
                        "embedding": embeddings[i]
                    })
                
                logger.info(f"✅ Added {len(chunks)} chunks with Sentence Transformers (384D) for {filename}")
            else:
                # Fallback: Use ChromaDB's default embeddings
                self.collection.add(
                    documents=chunks,
                    metadatas=metadatas,
                    ids=chunk_ids
                )
                
                # Store locally too
                for i, chunk in enumerate(chunks):
                    self.chunks.append({
                        "id": chunk_ids[i],
                        "content": chunk,
                        "filename": filename
                    })
                
                logger.info(f"✅ Added {len(chunks)} chunks with ChromaDB built-in embeddings for {filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            return False
    
    def search(self, query, k=3):
        """Search using Sentence Transformers or ChromaDB embeddings"""
        try:
            if self.use_sentence_transformers and self.sentence_model:
                # Advanced: Generate query embedding with Sentence Transformers
                query_embedding = self.sentence_model.encode([query])
                
                # Search in ChromaDB with custom embeddings
                results = self.collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=k,
                    include=['documents', 'metadatas', 'distances']
                )
                
                search_method = "Sentence Transformers (384D)"
            else:
                # Fallback: Use ChromaDB's default query (will embed automatically)
                results = self.collection.query(
                    query_texts=[query],
                    n_results=k,
                    include=['documents', 'metadatas', 'distances']
                )
                
                search_method = "ChromaDB built-in"
            
            # Process results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0]
                ):
                    similarity = 1 - distance  # Convert distance to similarity
                    search_results.append({
                        "content": doc,
                        "filename": metadata['filename'],
                        "similarity": float(similarity)
                    })
            
            logger.info(f"🔍 {search_method} search: {len(search_results)} results found")
            return search_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

# Initialize advanced RAG - ONLY modern AI
rag = AdvancedRAG()

# Simple document processing
def process_file(file_path, filename):
    """Simple, efficient file processing"""
    try:
        ext = filename.split('.')[-1].lower()
        
        if ext == 'pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return " ".join([page.extract_text() for page in reader.pages])
        
        elif ext == 'docx':
            doc = docx.Document(file_path)
            return " ".join([para.text for para in doc.paragraphs])
        
        elif ext in ['txt', 'md']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        return ""
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return ""

# Flask routes - Clean and simple
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'md'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Efficient upload processing"""
    try:
        files = request.files.getlist('files')
        processed = []
        
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                if filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
                    
                    # Save temporarily
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    
                    # Process with efficient method
                    content = process_file(file_path, filename)
                    if content and rag.add_document(filename, content):
                        processed.append(filename)
                    
                    # Cleanup
                    os.remove(file_path)
        
        if processed:
            return jsonify({
                'success': True,
                'message': f'Processed {len(processed)} files with Sentence Transformers + ChromaDB',
                'files': processed,
                'method': 'Advanced AI Only'
            })
        else:
            return jsonify({'error': 'No files processed'}), 400
            
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': "Sorry, I encountered an error while processing your question.",
            'method': 'Error fallback'
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Efficient chat with modern AI"""
    logger.info("📨 Chat request received")
    try:
        data = request.get_json()
        logger.info(f"📨 Request data: {data}")
        question = data.get('message', '').strip()
        logger.info(f"❓ Question: {question}")
        
        if not question:
            logger.warning("⚠️ No question provided")
            return jsonify({'error': 'No question provided'}), 400
        
        # Efficient search
        logger.info("🔍 Starting search...")
        results = rag.search(question)
        logger.info(f"🔍 Search completed, found {len(results)} results")
        
        if not results:
            logger.info("📭 No results found")
            return jsonify({
                'success': True,
                'response': "No relevant information found in documents.",
                'method': 'No results'
            })
        
        # Generate response with Gemini
        context = "\n".join([r["content"][:300] for r in results])
        prompt = f"""Based on this context from uploaded documents:

{context}

Question: {question}

Provide a clear, helpful answer based on the context:"""
        
        logger.info("🤖 Generating AI response...")
        try:
            if model:
                response = model.generate_content(prompt)
                answer = response.text
                logger.info(f"✅ Gemini response generated successfully")
            else:
                logger.warning("⚠️ No Gemini model available - using fallback")
                answer = f"Based on the documents: {results[0]['content'][:400]}..."
        except Exception as e:
            logger.error(f"❌ Gemini API error: {e}")
            answer = f"Based on the documents: {results[0]['content'][:400]}..."
        
        # Update conversation
        if 'conversation' not in session:
            session['conversation'] = []
        
        session['conversation'].append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'response': answer,
            'method': 'Sentence Transformers + ChromaDB + Gemini' if rag.use_sentence_transformers else 'ChromaDB + Gemini',
            'sources': [r['filename'] for r in results],
            'similarity': round(results[0]['similarity'], 3) if results else 0
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Chat processing failed'}), 500

@app.route('/clear', methods=['POST'])
def clear():
    """Clear conversation and documents - Advanced AI only"""
    session.clear()
    
    # Clear ChromaDB collection
    try:
        rag.collection.delete()
        rag.collection = rag.chroma_client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        rag.chunks = []
        logger.info("✅ Cleared ChromaDB and conversation")
    except Exception as e:
        logger.error(f"Clear error: {e}")
    
    return jsonify({'success': True, 'message': 'Cleared successfully'})

if __name__ == '__main__':
    logger.info("🚀 Advanced RAG starting - ONLY Sentence Transformers + ChromaDB!")
    app.run(debug=True, host='0.0.0.0', port=5000)
