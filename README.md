# Multi-Agent RAG Chatbot

[![Live Demo](https://img.shields.io/badge/🌐%20Live%20Demo-Click%20Here-blue?style=for-the-badge)](https://rag-chatbot-8ykv.onrender.com/)

## 🚀 Smart Document Q&A Bot

Professional RAG system with dual deployment strategy for development and production.

## ✨ Key Features

- **6 document formats** - PDF, DOCX, PPTX, CSV, TXT, MD
- **Voice input** with microphone 🎤  
- **Dual Architecture** - Advanced for dev, optimized for production
- **Multi-agent system** - 4 specialized AI agents
- **Real-time chat** interface
- **Sub-3 second responses**

## 🏗️ Dual Architecture

### **Local Development** (`flask_app/app.py`)
- **Sentence Transformers** - all-MiniLM-L6-v2 model (384d embeddings)
- **ChromaDB** - Persistent vector storage with HNSW indexing
- **Full AI Features** - Complete semantic understanding

### **Production Deployment** (`app.py`)
- **Optimized TF-IDF** - Enhanced with bigrams and better parameters
- **Lightweight** - Fits in 512MB Render memory limit  
- **Fast Startup** - No heavy model downloads

## ✨ Key Features

- **6 document formats** - PDF, DOCX, PPTX, CSV, TXT, MD
- **Voice input** with microphone 🎤  
- **Sentence Transformers** - Lightweight MiniLM embeddings (~80MB)
- **Multi-agent architecture** - 4 specialized AI agents
- **Real-time chat** interface
- **Sub-3 second responses**
- **Professional UI** with modern design

## 🧠 AI Technology

- **Sentence Transformers** - all-MiniLM-L6-v2 model (384d embeddings)
- **Semantic Search** - Cosine similarity with modern embeddings
- **Multi-Agent System** - Coordinator, Ingestion, Retrieval, LLM agents
- **Google Gemini** - AI response generation
- **Smart Chunking** - Intelligent document processing

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Add API key
echo "GEMINI_API_KEY=your_key_here" > .env

# Run app
python app.py

# Open browser
http://localhost:5000
```bash
# Set environment variables
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key

# Deploy automatically via GitHub
```

### **Local Development**
```bash
# Install dependencies
pip install -r requirements.txt
pip install sentence-transformers chromadb torch

# Add API keys to .env file
echo "GEMINI_API_KEY=your_gemini_key" > .env

# Run local version with full features
cd flask_app
python app.py

# Open browser
http://localhost:5000
```

## 💡 How It Works

1. **Upload documents** - Support for multiple formats
2. **Semantic processing** - Documents chunked and embedded using Sentence Transformers
3. **Vector storage** - Embeddings stored in ChromaDB for fast retrieval
4. **Ask questions** - Type or use voice input 🎤
5. **Semantic search** - Find relevant content using cosine similarity
6. **AI response** - Generate context-aware answers using Gemini AI

## 🏗️ Architecture

```
User Query → Semantic Embedding → Vector Search → Context Retrieval → AI Response
     ↓              ↓                    ↓              ↓              ↓
Voice/Text → SentenceTransformer → ChromaDB → RetrievalAgent → GeminiAI
```

## �️ Tech Stack

- **AI Models**: Google Gemini 1.5 Flash + Sentence Transformers
- **Vector DB**: ChromaDB with HNSW indexing
- **Backend**: Flask with multi-agent architecture
- **Frontend**: Modern HTML/CSS/JS with voice input
- **ML Libraries**: PyTorch, Transformers, SentenceTransformers
- **Document Processing**: PyPDF2, python-docx, python-pptx

## 📁 Project Structure

```
├── app.py                     # Main Flask application
├── agents/                    # Multi-agent system
│   ├── coordinator_agent.py  # Workflow orchestrator
│   ├── ingestion_agent.py    # Document processing
│   ├── retrieval_agent.py    # Vector search
│   └── llm_response_agent.py # AI responses
├── static/                    # Frontend assets
├── templates/                 # HTML templates
├── requirements.txt           # Dependencies
└── .env                      # API configuration
```

## 🎯 What Makes This Professional

### **Modern AI Stack**
- ✅ Sentence Transformers (not basic TF-IDF)
- ✅ ChromaDB vector database (not in-memory only)
- ✅ Semantic understanding (not keyword matching)
- ✅ Persistent storage (survives restarts)

### **Production Features**
- ✅ Multi-agent architecture
- ✅ Voice input integration
- ✅ Professional UI/UX
- ✅ Error handling & logging
- ✅ Session management
- ✅ Security best practices

### **Performance**
- ✅ Sub-3 second responses
- ✅ Efficient vector search
- ✅ Real-time similarity scoring
- ✅ Context-aware conversations

## 🔒 Security & Best Practices

- **API Key Protection**: Environment variables with `.env` file
- **Document Scope**: AI responses limited to uploaded content
- **Session Security**: Secure Flask session management
- **Input Validation**: Safe file handling and processing

## 🌐 Deployment

**Local Development:**
```bash
python app.py
# Access at http://localhost:5000
```

**Production Ready:**
- Professional Flask application
- Environment-based configuration
- Cloud deployment compatible (Render, Heroku, etc.)

## 📋 Example Usage

**For Resumes:**
- "What AI/ML skills are mentioned?"
- "What is the educational background?"
- "What programming languages are listed?"

**For Business Documents:**
- "What are the key findings?"
- "Can you summarize the main points?"
- "What recommendations are mentioned?"

---

**Built with modern AI technologies for enterprise-grade document Q&A.**
   ```
3. Open your browser to `http://localhost:5000`
4. Experience the professional interface with modern branding and UI

### Start Chatting!
6. Upload documents using the intuitive interface
7. Start asking questions about your uploaded documents!
8. **New**: Use the microphone button 🎤 for voice input - just click and speak!

## Enhanced Features & Capabilities

### 🎤 **Voice Input System**
- **One-Click Voice Input**: Click the microphone button and speak your question
- **Real-Time Transcription**: See your words appear as you speak
- **Smart Completion**: Automatic input field population with transcribed text
- **Visual Status Indicators**: Clear feedback for listening, processing, and completion states
- **Error Recovery**: Helpful messages for microphone permissions and browser compatibility

### 🧠 **Intelligent Response System**
- **Similarity-Based Filtering**: Questions are analyzed for relevance to uploaded documents
- **Adaptive Response Quality**: 
  - **High Relevance (40%+)**: Full AI-powered responses with context
  - **Medium Relevance (15-40%)**: Enhanced prompting with availability disclaimers
  - **Low Relevance (<15%)**: Helpful guidance and suggested questions
- **Document-Type Detection**: Smart suggestions based on content (resumes, reports, etc.)
- **Debug Transparency**: Real-time similarity scores for user understanding

### 🛡️ **Security & Safety Features**
- **Prompt Injection Protection**: Advanced safeguards against malicious inputs
- **Code Generation Prevention**: AI cannot be tricked into outputting programming code
- **Document-Only Responses**: Strict limitation to uploaded document content
- **Safe Fallback Messages**: Helpful guidance when queries fall outside document scope

## How it works
The system uses a sophisticated multi-agent architecture with **dual implementation strategies** for optimal performance:

### Production Architecture (Flask Interface)
- **CoordinatorAgent**: Orchestrates the entire workflow and manages agent communication
- **IngestionAgent**: Parses and extracts text from various document formats
- **RetrievalAgent**: Generates embeddings and performs semantic similarity search using ChromaDB
- **LLMResponseAgent**: Constructs context-aware prompts and generates AI responses

### Optimized Architecture (Streamlit Interface)
- **Inline MCP Implementation**: Lightweight message passing without heavy dependencies
- **Keyword-based Retrieval**: Fast text matching for instant responses (2-3 second startup)
- **Simple Document Processing**: Streamlined parsing for maximum speed
- **Direct LLM Integration**: Optimized Google Gemini API calls with rate limiting

All agents communicate through standardized MCP messages with trace IDs, ensuring reliable and traceable inter-agent communication while maintaining flexibility for different performance requirements.

## Project Structure
```
├── agents/                    # Core multi-agent processing logic
│   ├── coordinator_agent.py    # Main workflow orchestrator
│   ├── ingestion_agent.py      # Document parsing and chunking
│   ├── retrieval_agent.py      # Vector search and embedding
│   ├── llm_response_agent.py   # AI response generation with rate limiting
│   ├── mcp.py                  # Model Context Protocol implementation
│   ├── document_utils.py       # Multi-format document parsing utilities
│   └── embedding_utils.py      # ChromaDB and sentence transformers
├── ui/                        # Optimized Streamlit interface (2-3s startup)
│   └── app.py                 # Performance-optimized application with inline MCP
├── flask_app/                 # Professional production interface
│   ├── app.py                 # Flask web application
│   ├── templates/
│   │   └── index.html         # Professional HTML template with modern branding
│   └── static/
│       ├── css/styles.css     # Orange/black theme styling
│       └── js/app.js          # Interactive frontend JavaScript
├── .env                       # Secure API key configuration
├── requirements.txt           # Python dependencies
├── Presentation.pdf           # Project architecture and demo slides
└── README.md                  # Project documentation
```

## Sample Workflow (Message Passing with MCP)

**Example scenario:**
```
User uploads: sales_review.pdf, metrics.csv
User asks: "What KPIs were tracked in Q1?"
```

**Complete message flow through professional interface:**
```
➡️ Professional UI (Flask design) forwards to CoordinatorAgent
➡️ CoordinatorAgent orchestrates the pipeline:
   🔸 IngestionAgent → parses documents → sends CHUNKIFY_RESULT
   🔸 RetrievalAgent → finds relevant chunks → sends RETRIEVAL_RESULT  
   🔸 LLMResponseAgent → formats prompt & calls Gemini → sends LLM_ANSWER
➡️ Professional chatbot displays answer with source context in modern interface
```

**MCP message example:**
```json
{
  "type": "RETRIEVAL_RESULT",
  "sender": "RetrievalAgent", 
  "receiver": "LLMResponseAgent",
  "trace_id": "rag-457",
  "payload": {
    "retrieved_context": ["slide 3: revenue up", "doc: Q1 summary..."],
    "query": "What KPIs were tracked in Q1?"
  }
}
```

## Interface Features

### Professional Flask Interface
- **Modern branding** with orange/black corporate theme and professional logo
- **Voice Input Integration** with microphone button and real-time speech recognition
- **Modern UX** with drag-and-drop uploads, real-time chat, and responsive design
- **Smart Question Suggestions** based on document type (resume skills, technical experience, etc.)
- **Source context display** showing document chunks for answer verification
- **Similarity Score Indicators** for transparency in response relevance
- **Production-ready** with session management and error handling

### Development Interface (Streamlit) - Optimized for Speed
- **Lightning-fast startup** (2-3 seconds vs 30+ seconds previously)
- **Keyword-based retrieval** for instant document search
- **Inline MCP implementation** without heavy ML dependencies
- **Debug information** and expandable source context display
- **Performance monitoring** with trace IDs and processing metrics

## Usage Example
1. **Open the professional interface** at `http://localhost:5000`
2. **Upload documents** using drag-and-drop (supports PDF, DOCX, PPTX, CSV, TXT, MD)
3. **Ask questions**: 
   - **Text**: Type "What were the key financial highlights this year?"
   - **Voice**: Click 🎤 and speak "What AI skills are mentioned in this resume?"
4. **Get intelligent responses**: 
   - High-relevance questions get detailed AI-powered answers
   - Medium-relevance questions get enhanced responses with context
   - Low-relevance questions get helpful guidance and suggestions
5. **View transparency info**: See similarity scores and source context
6. **Continue conversations** naturally with context awareness

### Example Questions by Document Type:

**📄 For Resumes/CVs:**
- "What AI/ML skills does this person have?"
- "What is their educational background?"
- "What programming languages do they know?"
- "What work experience is listed?"

**📊 For Business Documents:**
- "What are the key findings?"
- "What recommendations are mentioned?"
- "What challenges were identified?"
- "Can you summarize the main points?"

## Advanced Features

### Dual Architecture Strategy
- **Production Mode**: Full vector embeddings with ChromaDB for maximum accuracy
- **Development Mode**: Optimized keyword matching for instant feedback during testing
- **Automatic fallbacks**: Intelligent rate limiting with offline response generation

### Model Context Protocol (MCP) Implementation
- **Structured messaging** with trace IDs for reliable agent communication
- **Flexible deployment**: Both full agent architecture and inline lightweight implementation
- **Error handling** and graceful failure recovery

### Multi-Agent Architecture
- **Separation of concerns** with specialized agents for parsing, retrieval, and response generation
- **Performance optimization** with dual implementation strategies
- **Scalable design** that can be extended with additional agents

### Performance Optimizations
- **Fast startup**: Streamlit interface loads in 2-3 seconds (vs 30+ seconds with heavy ML libraries)
- **Intelligent caching**: Document processing optimizations
- **Rate limit management**: Smart API usage with Google Gemini integration

## Notes
- **Internet Connection**: Required for Google Gemini API calls
- **Performance**: Streamlit interface optimized for speed (2-3s startup), Flask interface optimized for production features
- **Document Processing**: Larger documents may take a moment to process in production mode
- **Interface Selection**: 
  - **Streamlit** (`localhost:8501`): Fast development and testing
  - **Flask** (`localhost:5000`): Professional demonstrations and production use

## Security & Best Practices
- **API Key Protection**: This project uses environment variables to securely store API keys
- **Never commit API keys**: The `.env` file is excluded from version control via `.gitignore`
- **Environment Setup**: Always use the `.env` file for sensitive configuration
- **Production Deployment**: Use proper secret management in production environments
- **Session Security**: Flask sessions are configured with secure secret keys
- **File Upload Security**: Proper file validation and secure filename handling

## Deployment Options

### Local Development
- **Streamlit**: `streamlit run ui/app.py` (Port 8501)
- **Flask**: `python flask_app/app.py` (Port 5000)

### Production Deployment
The Flask application is production-ready with:
- **Professional branding** for corporate environments
- **Secure file handling** and session management
- **Error handling** and logging for monitoring
- **Responsive design** for various devices
- **Professional UI/UX** suitable for client demonstrations

## Project Highlights

### 🏆 **Advanced AI Engineering**
- **Multi-agent architecture** with specialized responsibilities and dual implementation strategies
- **Model Context Protocol** for reliable inter-agent communication with trace IDs
- **Enhanced Similarity Processing**: Optimized cosine distance calculation with 15% relevance threshold
- **Flexible retrieval systems**: Vector search (production) + keyword matching (development)
- **Multiple document formats** with robust parsing and error handling
- **Performance optimization**: 93% startup time reduction (30s → 2-3s)
- **Smart Response Strategies**: Adaptive AI responses based on content relevance levels

### 🎨 **Professional Design & UX**
- **Modern branding** with authentic company colors and logo
- **Voice Input Integration**: Hands-free operation with Web Speech API
- **Dual interface strategy**: Development (Streamlit) + Production (Flask)
- **Modern UI/UX** with drag-and-drop functionality and real-time chat
- **Smart Suggestions**: Context-aware question prompts based on document type
- **Responsive design** optimized for all devices
- **Professional typography** and visual hierarchy

### 🔒 **Security & Quality**
- **Environment variable configuration** for secure API key management
- **Advanced Prompt Engineering**: Protection against code generation and prompt injection
- **Similarity-based filtering**: Prevents irrelevant or potentially unsafe responses
- **Document-scope limitation**: AI responses strictly based on uploaded content
- **Comprehensive error handling** with graceful degradation
- **Session management** for persistent conversations
- **Professional logging** and monitoring capabilities

## Recent Changes & Improvements Summary

### 🎯 **August 2025 Comprehensive Enhancement Overview**

This project has undergone significant improvements to enhance reliability, user experience, and security. Here's what's been implemented:

#### 🎤 **Voice Input System**
- **Web Speech API Integration**: Full browser-based speech recognition with real-time feedback
- **Voice Status Indicators**: Visual feedback showing listening state and transcription progress  
- **Cross-browser Compatibility**: Optimized for Chrome, Edge, Safari, and Firefox
- **Error Handling**: Graceful fallback when voice features are unavailable
- **Accessibility**: Alternative input method for users who prefer hands-free interaction

#### 🛡️ **Enhanced Security & Prompt Engineering**
- **Advanced Prompt Safety**: Multi-layer protection against prompt injection attacks
- **Document-focused Responses**: AI strictly limited to uploaded document content
- **Code Generation Prevention**: Safety mechanisms prevent potentially harmful code output
- **Similarity-based Filtering**: 15% relevance threshold ensures contextually appropriate responses
- **API Key Security**: Environment variable configuration for secure credential management

#### 🎨 **User Experience Improvements**
- **Smart Question Suggestions**: Context-aware prompts based on detected document type
- **Real-time Status Updates**: Loading indicators and processing feedback
- **Enhanced Error Messages**: Clear, actionable feedback for users
- **Professional UI Polish**: Orange/black branding with modern design principles
- **Responsive Voice Controls**: Intuitive microphone button with visual state changes

#### ⚙️ **Technical Improvements**
- **Optimized Similarity Calculations**: Fixed cosine distance conversion for accurate retrieval
- **Enhanced Debug Logging**: Comprehensive tracking for troubleshooting and optimization
- **Multi-level Response Strategies**: Adaptive AI behavior based on content relevance
- **Improved PDF Processing**: Robust document parsing with better chunk extraction
- **Performance Monitoring**: Detailed similarity score tracking and threshold optimization

#### 🧪 **Testing & Validation**
- **Comprehensive Testing Suite**: Dedicated test scripts for similarity calculations and PDF processing
- **Real-world Validation**: Tested with actual documents achieving 35-36% similarity scores
- **Voice Input Testing**: Verified functionality across multiple browsers and environments
- **End-to-end Validation**: Complete workflow testing from document upload to AI response

---

**This project features professional design standards for modern user experience.**
