# -*- coding: utf-8 -*-
import sys
import io
import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import time
from functools import wraps

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Ensure agents can be imported correctly
sys.path.insert(0, os.path.dirname(__file__))

# Multi-Agent Architecture with MCP Protocol
from agents.coordinator_agent import CoordinatorAgent
from agents.embedding_utils import EmbeddingStore
from agents.report_generator import ClinicalReportGenerator

load_dotenv()

from flask_sqlalchemy import SQLAlchemy
import uuid
import json
import secrets

# Configure logging first to enable warning logs during initialization
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve allowed CORS origins from environment variable (comma-separated string)
allowed_origins_env = os.environ.get('ALLOWED_ORIGINS')
if allowed_origins_env:
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(',') if origin.strip()]
else:
    allowed_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

logger.info(f"🔒 Configured Allowed CORS Origins: {allowed_origins}")

app = Flask(__name__, static_folder=None)

# Apply ProxyFix middleware to trust Render's reverse proxy headers (1 hop)
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

CORS(app, supports_credentials=True, origins=allowed_origins)

# Initialize Flask-Limiter for rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_limiter.errors import RateLimitExceeded

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],
    # Note: memory:// store is intentional given the current single-worker deployment.
    # If scaled to multiple Gunicorn workers or instances, this must be switched to redis://.
    storage_uri="memory://"
)

@app.errorhandler(RateLimitExceeded)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded, please slow down"}), 429

# Lightweight request timing decorator
def time_request(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            response = func(*args, **kwargs)
            return response
        finally:
            duration = time.perf_counter() - start_time
            logger.info(f"⏱️ {request.path} completed in {duration:.2f}s")
    return wrapper

env_secret = os.environ.get('SECRET_KEY')
if env_secret:
    app.secret_key = env_secret
else:
    app.secret_key = secrets.token_hex(32)
    logger.warning("⚠️ WARNING: SECRET_KEY environment variable is not set. Generated a random secret key. User sessions will not persist across server restarts!")

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB for medical reports
app.config['ENV'] = os.environ.get('FLASK_ENV', 'development')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# Database Configuration: Use DATABASE_URL if provided, else fall back to local SQLite
db_url = os.environ.get('DATABASE_URL')
if db_url:
    # Handle Render/Heroku legacy postgres:// schema mapping to postgresql://
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
    logger.info("🗄️ Database: Configured to use external SQL Database from environment.")
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nutrimind.db'
    logger.info("🗄️ Database: Configured to use local SQLite database (nutrimind.db).")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class UserSession(db.Model):
    id = db.Column(db.String, primary_key=True)  # UUID
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    profile_json = db.Column(db.Text)  # serialized Profile
    uploaded_files_json = db.Column(db.Text)  # serialized list of uploaded file names
    meal_plan_json = db.Column(db.Text)      # serialized meal plan
    training_plan_json = db.Column(db.Text)  # serialized training split
    bio_age_json = db.Column(db.Text)        # serialized biological age metrics
    critique_json = db.Column(db.Text)       # serialized board review critique
    audit_report = db.Column(db.Text)        # raw safety audit report text
    corrections_json = db.Column(db.Text)    # serialized list of safety corrections

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(db.String, db.ForeignKey('user_session.id'), nullable=False)
    role = db.Column(db.String, nullable=False)  # 'user' | 'assistant'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    try:
        # Detect if database has new schema columns
        db.session.execute(db.text("SELECT meal_plan_json FROM user_session LIMIT 1"))
    except Exception:
        # Drop and recreate if older schema version is loaded
        logger.info("Schema mismatch detected in SQLite. Dropping and re-initializing database tables...")
        db.drop_all()
    db.create_all()

def get_or_create_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session.modified = True
    return session['session_id']

# ==============================================================================
# 🚀 SCALING ROADMAP & DEPLOYMENT ARCHITECTURE BOUNDARIES (Deliberate Design)
# ==============================================================================
# This application is intentionally deployed with `gunicorn --workers 1` because:
#
# 1. Process-Local Singletons:
#    - `EmbeddingStore` (ChromaDB client) runs in-process. Multiple Gunicorn workers
#      would spawn separate, conflicting instances of the vector database engine.
#    - `_local_pipeline_lock` (PyTorch CPU generation mutex) is a process-local
#      threading.Lock(). Multiple workers would violate this mutex boundary, causing
#      concurrent CPU-bound model invocations and severe core thrashing.
#
# 2. Ephemeral In-Memory State:
#    - Flask-Limiter is configured with `storage_uri="memory://"`.
#    - Flask `SECRET_KEY` falls back to a random startup token if environment is unset.
#    Both of these require a single-worker architecture to maintain session continuity
#    and rate limiter consistency.
#
# Horizontal Scaling Checklist:
# To scale this application beyond a single worker or container instance, you must:
#   - Migrating Locks: Replace `_local_pipeline_lock` with a Redis-backed distributed lock.
#   - Shared Limiter: Bind `Flask-Limiter` to a shared Redis service (`storage_uri="redis://..."`).
#   - Session Persistence: Enforce a static, persistent `SECRET_KEY` environment variable.
#   - Centralized Vector Search: Switch ChromaDB from in-memory/local sqlite to a hosted Vector DB (e.g. Pinecone/Qdrant).
# ==============================================================================

# Native Thread Pool for Asynchronous Background Ingestion and Parallel Executions
bg_executor = ThreadPoolExecutor(max_workers=4)

# Initialize the Multi-Agent RAG system
class AgenticRAG:
    """Multi-Agent RAG with ChromaDB + Advanced Health Orchestrator"""

    def __init__(self):
        self.coordinator = CoordinatorAgent()
        self.embedding_store = EmbeddingStore(persist_path="./chroma_advanced_v2")
        self.coordinator.retrieval_agent.vector_store = self.embedding_store
        
        self.file_paths = []  # Track uploaded file paths
        logger.info("✅ Advanced NutriMind RAG Agent System initialized!")

    def ingest_document_foreground(self, file_path, filename):
        """Parse health report, extract clinical markers (ClinicalAnalyzerAgent) immediately"""
        analysis = self.coordinator.analyze_document([file_path])
        
        if "error" in analysis:
            logger.error(f"Error analyzing document: {analysis['error']}")
            return {"success": False, "error": analysis["error"], "mcp_trace": analysis.get("mcp_trace", [])}
            
        chunks = analysis.get("chunks", [])
        profile = analysis.get("profile", {})
        
        return {
            "success": True,
            "profile": profile,
            "chunks": chunks,
            "mcp_trace": analysis.get("mcp_trace", [])
        }

    def query(self, question, profile):
        """Run Plan-Reason-Audit multi-agent query loop with parallel vector context retrieval"""
        # Under the hood, the CoordinatorAgent.process_health_query now executes the ChromaDB search
        # and PubMed/UpToDate web research in parallel to minimize latency!
        result = self.coordinator.process_health_query(question, profile)
        return result

    def clear(self):
        """Clear all data and reset agents"""
        self.embedding_store.clear()
        self.file_paths = []
        logger.info("🗑️ All agent data and ChromaDB collection cleared")

rag = AgenticRAG()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'pptx', 'csv', 'txt', 'md'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def async_background_embedding(chunks, filename):
    """Background thread function that indexes chunks in ChromaDB asynchronously"""
    try:
        logger.info(f"⚡ Starting async text chunk embedding for {filename} ({len(chunks)} chunks)...")
        rag.embedding_store.add_chunks(chunks)
        logger.info(f"✅ Async embedding completed successfully for {filename}!")
    except Exception as e:
        logger.error(f"❌ Error in async embedding for {filename}: {e}")

@app.route('/api/upload', methods=['POST'])
@limiter.limit("10 per minute")
@time_request
def upload():
    """Upload health reports → extract biomarkers in the foreground (fast) and index chunks in background"""
    try:
        # Request Validation Guards
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
            
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No valid files selected'}), 400
            
        for file in files:
            if file and file.filename:
                ext = file.filename.rsplit('.', 1)[-1].lower()
                if ext not in ALLOWED_EXTENSIONS:
                    return jsonify({'error': f'File type not allowed: {ext}'}), 400
                    
                file.seek(0, os.SEEK_END)
                size = file.tell()
                file.seek(0)
                if size == 0:
                    return jsonify({'error': f'File is empty: {file.filename}'}), 400
                if size > 20 * 1024 * 1024:
                    return jsonify({'error': f'File too large: {file.filename} (max 20MB per file)'}), 400

        logger.info("🗑️ Clearing previous data before new upload...")
        session_id = get_or_create_session_id()
        # Delete existing session and messages in DB for this UUID
        ChatMessage.query.filter_by(session_id=session_id).delete()
        UserSession.query.filter_by(id=session_id).delete()
        db.session.commit()
        rag.clear()
        processed = []
        failed = []
        extracted_profile = None
        mcp_trace = []

        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                if filename.rsplit('.', 1)[-1].lower() in ALLOWED_EXTENSIONS:
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)

                    # Ingest and analyze health records (Foreground profile extraction)
                    ingest_result = rag.ingest_document_foreground(file_path, filename)
                    
                    if ingest_result.get("success"):
                        processed.append(filename)
                        if "failed_files" in ingest_result:
                            failed.extend(ingest_result["failed_files"])
                        extracted_profile = ingest_result.get("profile")
                        mcp_trace.extend(ingest_result.get("mcp_trace", []))
                        
                        # Fetch text chunks and dispatch to background thread pool for embedding
                        chunks = ingest_result.get("chunks", [])
                        if chunks:
                            bg_executor.submit(async_background_embedding, chunks, filename)
                    
                    # Cleanup temp file
                    try:
                        os.remove(file_path)
                    except:
                        pass

        if processed:
            if not extracted_profile:
                logger.warning("⚠️ Clinical profile extraction returned empty or null. Using default baseline profile.")
                extracted_profile = {
                    "demographics": {"age": 30, "weight_kg": 70, "height_cm": 170, "gender": "Male", "activity_level": "Moderate"},
                    "goals": ["General healthy living"],
                    "allergies": [],
                    "medical_conditions": [],
                    "biomarkers": []
                }
                
            user_sess = UserSession.query.filter_by(id=session_id).first()
            if not user_sess:
                user_sess = UserSession(
                    id=session_id,
                    profile_json=json.dumps(extracted_profile),
                    uploaded_files_json=json.dumps(processed)
                )
                db.session.add(user_sess)
            else:
                user_sess.profile_json = json.dumps(extracted_profile)
                user_sess.uploaded_files_json = json.dumps(processed)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': f'Successfully ingested {len(processed)} files. Demographics and biomarkers extracted in foreground. Document index populating asynchronously.',
                'files': processed,
                'profile': extracted_profile,
                'mcp_trace': mcp_trace
            })
        else:
            return jsonify({'error': 'Failed to extract biomarkers or no valid files processed'}), 400

    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'method': 'Error in Ingestion/Analysis'
        }), 500

@app.route('/api/chat', methods=['POST'])
@limiter.limit("20 per minute")
@time_request
def chat():
    """Chat → Run Plan-Reason-Audit multi-agent loop with native concurrency optimizations"""
    try:
        # Request Validation Guards
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "message field required"}), 400
        question = data.get('message')
        if not isinstance(question, str):
            return jsonify({"error": "message must be a string"}), 400
        question = question.strip()
        if not question:
            return jsonify({"error": "message cannot be empty"}), 400
        if len(question) > 4000:
            return jsonify({"error": "message too long (max 4000 characters)"}), 400

        # Retrieve biomarker profile from SQLite
        session_id = get_or_create_session_id()
        user_sess = UserSession.query.filter_by(id=session_id).first()
        profile = None
        if user_sess and user_sess.profile_json:
            profile = json.loads(user_sess.profile_json)

        if not profile:
            # Generate a default healthy baseline profile if user has not uploaded files yet
            profile = {
                "demographics": {"age": 30, "weight_kg": 70, "height_cm": 170, "gender": "Male", "activity_level": "Moderate"},
                "goals": ["General healthy living"],
                "allergies": [],
                "medical_conditions": [],
                "biomarkers": []
            }
            logger.info("No clinical profile in database. Using default healthy baseline.")
            
        # Ensure a parent UserSession row always exists in SQLite before adding chat messages (FOREIGN KEY safety)
        if not user_sess:
            user_sess = UserSession(
                id=session_id,
                profile_json=json.dumps(profile),
                uploaded_files_json=json.dumps([])
            )
            db.session.add(user_sess)
            db.session.commit()

        # Run the full health agent pipeline via Coordinator (now parallelized)
        result = rag.query(question, profile)
        
        # Save structured results to SQLite session
        if user_sess:
            user_sess.meal_plan_json = json.dumps(result.get('meal_plan'))
            user_sess.training_plan_json = json.dumps(result.get('training_plan'))
            user_sess.bio_age_json = json.dumps(result.get('bio_age_results'))
            user_sess.critique_json = json.dumps(result.get('critique'))
            user_sess.audit_report = result.get('audit_report')
            user_sess.corrections_json = json.dumps(result.get('corrections', []))
        
        # Save conversation messages to SQLite
        user_msg = ChatMessage(session_id=session_id, role='user', content=question)
        assistant_msg = ChatMessage(session_id=session_id, role='assistant', content=result.get('answer'))
        db.session.add(user_msg)
        db.session.add(assistant_msg)
        db.session.commit()

        return jsonify({
            'success': True,
            'response': result.get('answer'),
            'meal_plan': result.get('meal_plan'),
            'training_plan': result.get('training_plan'),
            'targets': result.get('targets'),
            'audit_report': result.get('audit_report'),
            'corrections': result.get('corrections', []),
            'bio_age_results': result.get('bio_age_results'),
            'critique': result.get('critique'),
            'mcp_trace': result.get('mcp_trace', [])
        })

    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        return jsonify({'error': f'Chat processing failed: {str(e)}'}), 500

@app.route('/api/clear', methods=['POST'])
def clear():
    """Clear all agent data and conversation"""
    try:
        session_id = session.get('session_id')
        if session_id:
            ChatMessage.query.filter_by(session_id=session_id).delete()
            UserSession.query.filter_by(id=session_id).delete()
            db.session.commit()
        session.clear()
        rag.clear()
        return jsonify({
            'success': True,
            'message': 'All clinical profiles, collections, and conversation history cleared'
        })
    except Exception as e:
        logger.error(f"❌ Clear error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to clear: {str(e)}'
        }), 500

@app.route('/api/report/download', methods=['GET'])
def download_report():
    """Download compiled high-fidelity clinical PDF report based on current session data"""
    try:
        from flask import send_file
        session_id = get_or_create_session_id()
        user_sess = UserSession.query.filter_by(id=session_id).first()
        if not user_sess or not user_sess.profile_json:
            return jsonify({'error': 'No active clinical profile found. Please upload a report or query the agents first.'}), 400
            
        profile = json.loads(user_sess.profile_json)
        
        # Deserialize current plan structures if present
        meal_plan = json.loads(user_sess.meal_plan_json) if user_sess.meal_plan_json else None
        training_plan = json.loads(user_sess.training_plan_json) if user_sess.training_plan_json else None
        bio_age_results = json.loads(user_sess.bio_age_json) if user_sess.bio_age_json else None
        critique = json.loads(user_sess.critique_json) if user_sess.critique_json else None
        audit_report = user_sess.audit_report
        corrections = json.loads(user_sess.corrections_json) if user_sess.corrections_json else []
        
        # Compile PDF report using ReportLab
        pdf_bytes = ClinicalReportGenerator.generate_pdf(
            profile=profile,
            meal_plan=meal_plan,
            training_plan=training_plan,
            bio_age_results=bio_age_results,
            critique=critique,
            audit_report=audit_report,
            corrections=corrections
        )
        
        # Serve bytes as file attachment in-memory
        pdf_buffer = io.BytesIO(pdf_bytes)
        pdf_buffer.seek(0)
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='NutriMind_Clinical_Report.pdf'
        )
    except Exception as e:
        logger.error(f"❌ PDF generation failed: {e}")
        return jsonify({'error': f'PDF report generation failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check with agent status and database model details"""
    session_id = session.get('session_id')
    has_profile = False
    profile = None
    if session_id:
        user_sess = UserSession.query.filter_by(id=session_id).first()
        if user_sess and user_sess.profile_json:
            has_profile = True
            profile = json.loads(user_sess.profile_json)

    return jsonify({
        'status': 'healthy',
        'app': 'NutriMind AI: Next-Gen Agentic Health & Diet Companion',
        'agents': {
            'clinical_analyzer': 'active',
            'web_researcher': 'active',
            'nutri_planner': 'active',
            'safety_auditor': 'active',
            'coordinator': 'active'
        },
        'vector_db': f"ChromaDB + HNSW (Model: {rag.embedding_store.model_name})",
        'has_profile': has_profile,
        'profile': profile
    })

# Serve built React app on any non-API route (Vite SPA support)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'dist')
    logger.info(f"DEBUG PATH: path='{path}', static_dir='{static_dir}', exists={os.path.exists(os.path.join(static_dir, 'index.html'))}")
    if path != "" and os.path.exists(os.path.join(static_dir, path)):
        return send_from_directory(static_dir, path)
    else:
        return send_from_directory(static_dir, 'index.html')

if __name__ == '__main__':
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting NutriMind Agentic Health Coach on {host}:{port}!")
    logger.info(f"⚙️ FLASK DEBUG MODE: {app.config['DEBUG']}")
    app.run(debug=app.config['DEBUG'], host=host, port=port)
