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
  - **Streamlit**: Development and testing interface
  - **Flask**: Production web application with professional SLRIS-inspired branding
  - **HTML/CSS/JavaScript**: Custom responsive design with corporate orange/black theme
- **AI Engine**: Google Gemini 1.5 Flash (Large Language Model)
- **Vector Search**: ChromaDB with Sentence Transformers embeddings
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
The system uses a sophisticated multi-agent architecture where different specialized components handle specific tasks through structured Model Context Protocol (MCP) messaging:

- **CoordinatorAgent**: Orchestrates the entire workflow and manages agent communication
- **IngestionAgent**: Parses and extracts text from various document formats
- **RetrievalAgent**: Generates embeddings and performs semantic similarity search
- **LLMResponseAgent**: Constructs context-aware prompts and generates AI responses

All agents communicate through standardized MCP messages, ensuring reliable and traceable inter-agent communication.

## Project Structure
```
├── agents/                    # Core multi-agent processing logic
│   ├── coordinator_agent.py    # Main workflow orchestrator
│   ├── ingestion_agent.py      # Document parsing and chunking
│   ├── retrieval_agent.py      # Vector search and embedding
│   ├── llm_response_agent.py   # AI response generation
│   ├── mcp.py                  # Model Context Protocol implementation
│   ├── document_utils.py       # Multi-format document parsing
│   ├── embedding_utils.py      # ChromaDB and sentence transformers
│   └── llm_utils.py           # Google Gemini API integration
├── ui/                        # Streamlit development interface
│   └── app.py                 # Main Streamlit application
├── flask_app/                 # Professional production interface
│   ├── app.py                 # Flask web application (SLRIS-inspired)
│   ├── templates/
│   │   └── index.html         # Professional HTML template with SLRIS branding
│   └── static/
│       ├── css/styles.css     # SLRIS orange/black theme styling
│       └── js/app.js          # Interactive frontend JavaScript
├── .env                       # Secure API key configuration
├── requirements.txt           # Python dependencies
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

### Development Interface (Streamlit)
- **Rapid prototyping** for testing and development
- **Debug information** and expandable source context display

## Usage Example
1. **Open the professional interface** at `http://localhost:5000`
2. **Upload documents** using drag-and-drop
3. **Ask questions**: "What were the key financial highlights this year?"
4. **Get AI-powered answers** with source references
5. **Continue the conversation** naturally

## Advanced Features

### Model Context Protocol (MCP) Implementation
- **Structured messaging** with trace IDs for reliable agent communication
- **Error handling** and graceful failure recovery

### Multi-Agent Architecture
- **Separation of concerns** with specialized agents for parsing, retrieval, and response generation
- **Scalable design** that can be extended with additional agents

## Notes
- **Internet Connection**: Required for Google Gemini API calls
- **Document Processing**: Larger documents may take a moment to process
- **Professional Interface**: Flask interface recommended for demonstrations and production

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
- **Multi-agent architecture** with specialized responsibilities
- **Model Context Protocol** for reliable inter-agent communication
- **Vector search** with semantic similarity using sentence transformers
- **Multiple document formats** with robust parsing

### 🎨 **Professional Design**
- **SLRIS-inspired branding** with authentic company colors and logo
- **Modern UI/UX** with drag-and-drop functionality
- **Responsive design** optimized for all devices
- **Professional typography** and visual hierarchy

### 🔒 **Security & Quality**
- **Environment variable configuration** for secure API key management
- **Comprehensive error handling** with graceful degradation
- **Session management** for persistent conversations
- **Professional logging** and monitoring capabilities

---

**This project was visually and structurally inspired by SLRIS's standards for professional design and user experience.**
