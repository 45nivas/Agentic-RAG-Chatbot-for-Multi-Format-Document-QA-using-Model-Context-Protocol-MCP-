# Agentic RAG Chatbot for Multi-Format Document QA
## Smart Document Q&A Bot with SLRIS-Inspired Professional Interface

An intelligent multi-agent chatbot that can answer questions about your documents using AI and Model Context Protocol (MCP). Upload PDFs, Word docs, PowerPoints and more - then ask questions and get instant answers through a professionally designed interface inspired by SLRIS company branding!

## What it does
- **Multi-format document support**: PDF, DOCX, PPTX, CSV, TXT, Markdown
- **Natural language queries**: Ask questions in plain English
- **AI-powered responses**: Get intelligent answers with source references
- **Multi-turn conversations**: Context-aware chat that remembers previous questions
- **Professional dual interface**: 
  - **Development UI**: Clean Streamlit interface for testing
  - **Production UI**: Professional Flask web app with SLRIS-inspired design featuring authentic orange/black branding and corporate logo

## Tech Stack
- **Frontend & UI**: 
  - **Streamlit**: Optimized development interface with fast loading (2-3 seconds)
  - **Flask**: Production web application with professional SLRIS-inspired branding
  - **HTML/CSS/JavaScript**: Custom responsive design with corporate orange/black theme
- **AI Engine**: Google Gemini 1.5 Flash (Large Language Model) with intelligent rate limiting
- **Search System**: 
  - **Production**: ChromaDB with Sentence Transformers embeddings for vector search
  - **Optimized**: Keyword-based retrieval for lightning-fast performance
- **Document Processing**: PyPDF2, python-docx, python-pptx, pandas
- **Architecture**: Multi-agent system with Model Context Protocol (MCP)
- **Security**: Environment variable configuration with python-dotenv
- **Branding**: Professional SLRIS company-inspired design with authentic logo integration

## Quick Start

### Option 1: Streamlit Development Interface
1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up your API key securely:**
   ```bash
   # Create a .env file in the project root
   echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
   ```
   Replace `your_actual_api_key_here` with your Google Gemini API key.

4. Run the Streamlit application:
   ```bash
   streamlit run ui/app.py
   ```
5. Open your browser to `http://localhost:8501`

### Option 2: Professional Flask Interface (SLRIS-Inspired)
1. Follow steps 1-3 above for setup
2. Run the Flask application:
   ```bash
   python flask_app/app.py
   ```
3. Open your browser to `http://localhost:5000`
4. Experience the professional SLRIS-inspired interface with authentic branding and modern UI

### Start Chatting!
6. Upload documents using the intuitive interface
7. Start asking questions about your uploaded documents!

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
│   ├── app.py                 # Flask web application (SLRIS-inspired)
│   ├── templates/
│   │   └── index.html         # Professional HTML template with SLRIS branding
│   └── static/
│       ├── css/styles.css     # SLRIS orange/black theme styling
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

**Complete message flow through professional SLRIS-inspired interface:**
```
➡️ Professional UI (Flask/SLRIS design) forwards to CoordinatorAgent
➡️ CoordinatorAgent orchestrates the pipeline:
   🔸 IngestionAgent → parses documents → sends CHUNKIFY_RESULT
   🔸 RetrievalAgent → finds relevant chunks → sends RETRIEVAL_RESULT  
   🔸 LLMResponseAgent → formats prompt & calls Gemini → sends LLM_ANSWER
➡️ Professional chatbot displays answer with source context in SLRIS-branded interface
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

### Professional Flask Interface (SLRIS-Inspired)
- **Authentic SLRIS branding** with orange/black corporate theme and real company logo
- **Modern UX** with drag-and-drop uploads, real-time chat, and responsive design
- **Source context display** showing document chunks for answer verification
- **Production-ready** with session management and error handling

### Development Interface (Streamlit) - Optimized for Speed
- **Lightning-fast startup** (2-3 seconds vs 30+ seconds previously)
- **Keyword-based retrieval** for instant document search
- **Inline MCP implementation** without heavy ML dependencies
- **Debug information** and expandable source context display
- **Performance monitoring** with trace IDs and processing metrics

## Usage Example
1. **Open the professional interface** at `http://localhost:5000`
2. **Upload documents** using drag-and-drop
3. **Ask questions**: "What were the key financial highlights this year?"
4. **Get AI-powered answers** with source references
5. **Continue the conversation** naturally

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
- **Professional SLRIS branding** for corporate environments
- **Secure file handling** and session management
- **Error handling** and logging for monitoring
- **Responsive design** for various devices
- **Professional UI/UX** suitable for client demonstrations

## Project Highlights

### 🏆 **Advanced AI Engineering**
- **Multi-agent architecture** with specialized responsibilities and dual implementation strategies
- **Model Context Protocol** for reliable inter-agent communication with trace IDs
- **Flexible retrieval systems**: Vector search (production) + keyword matching (development)
- **Multiple document formats** with robust parsing and error handling
- **Performance optimization**: 93% startup time reduction (30s → 2-3s)

### 🎨 **Professional Design**
- **SLRIS-inspired branding** with authentic company colors and logo
- **Dual interface strategy**: Development (Streamlit) + Production (Flask)
- **Modern UI/UX** with drag-and-drop functionality and real-time chat
- **Responsive design** optimized for all devices
- **Professional typography** and visual hierarchy

### 🔒 **Security & Quality**
- **Environment variable configuration** for secure API key management
- **Comprehensive error handling** with graceful degradation
- **Session management** for persistent conversations
- **Professional logging** and monitoring capabilities

---

**This project was visually and structurally inspired by SLRIS's standards for professional design and user experience.**
