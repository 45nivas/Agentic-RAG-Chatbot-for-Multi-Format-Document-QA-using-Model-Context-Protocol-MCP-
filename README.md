# Agentic RAG Chatbot for Multi-Format Document QA
## Smart Document Q&A Bot with Professional Interface

An intelligent multi-agent chatbot that can answer questions about your documents using AI and Model Context Protocol (MCP). Upload PDFs, Word docs, PowerPoints and more - then ask questions and get instant answers through a professionally designed interface!

## What it does
- **Multi-format document support**: PDF, DOCX, PPTX, CSV, TXT, Markdown
- **Natural language queries**: Ask questions in plain English
- **AI-powered responses**: Get intelligent answers with source references
- **Multi-turn conversations**: Context-aware chat that remembers previous questions
- **Voice-to-Text Input**: Browser-based speech recognition using Web Speech API
- **Smart Similarity Filtering**: Enhanced prompt engineering with similarity thresholds
- **Professional dual interface**: 
  - **Development UI**: Clean Streamlit interface for testing
  - **Production UI**: Professional Flask web app with modern design featuring elegant orange/black branding

## Recent Enhancements (August 2025)

### 🎤 **Voice Input & Speech Recognition**
- **Microphone Button**: Added adjacent to input field for hands-free operation
- **Web Speech API Integration**: Browser-based speech-to-text with real-time transcription
- **Visual Feedback**: Recording animations, status indicators, and completion notifications
- **Error Handling**: Comprehensive support for unsupported browsers and permission issues
- **Cross-Browser Support**: Works on Chrome, Edge, and Chromium-based browsers

### 🛡️ **Enhanced Security & Prompt Engineering**
- **Similarity Threshold Filtering**: Questions below 15% relevance show helpful guidance instead of potentially unsafe responses
- **Secure LLM Prompting**: Advanced prompt engineering prevents code generation and system command execution
- **Document-Focused Responses**: AI strictly limited to answering based on uploaded document content
- **Protection Against Prompt Injection**: Safeguards against attempts to bypass system restrictions

### 🎯 **Smart User Experience**
- **Context-Aware Suggestions**: Auto-detects resume/CV documents and provides relevant question suggestions
- **Dynamic Question Prompts**: Tailored suggestions based on document type (resume skills, experience, education)
- **Debug Information**: Real-time similarity scores and relevance indicators for transparency
- **Enhanced Error Messages**: Helpful guidance when questions don't match document content
- **Progressive Response Quality**: Different response strategies based on similarity confidence levels

### 🔧 **Technical Improvements**
- **Optimized Similarity Calculation**: Fixed cosine distance computation with ChromaDB integration
- **Enhanced Embedding Configuration**: Proper cosine similarity space configuration for better accuracy
- **Improved Document Parsing**: Better text extraction and chunking for various file formats
- **Real-time Debug Logging**: Comprehensive debugging information for similarity scores and document matching
- **Multi-level Response Strategy**: Adaptive responses based on content relevance (15-40% vs 40%+ similarity)

## Tech Stack
- **Frontend & UI**: 
  - **Streamlit**: Optimized development interface with fast loading (2-3 seconds)
  - **Flask**: Production web application with professional branding and voice input
  - **HTML/CSS/JavaScript**: Custom responsive design with corporate orange/black theme
  - **Web Speech API**: Browser-based voice recognition for hands-free interaction
- **AI Engine**: Google Gemini 1.5 Flash (Large Language Model) with intelligent rate limiting and enhanced prompting
- **Search System**: 
  - **Production**: ChromaDB with Sentence Transformers embeddings for vector search and cosine similarity
  - **Optimized**: Keyword-based retrieval for lightning-fast performance
  - **Smart Filtering**: Similarity threshold-based response filtering (15% minimum relevance)
- **Document Processing**: PyPDF2, python-docx, python-pptx, pandas with enhanced text extraction
- **Architecture**: Multi-agent system with Model Context Protocol (MCP) and debug logging
- **Security**: Environment variable configuration with python-dotenv and prompt injection protection
- **Branding**: Professional design with modern logo integration and voice interaction indicators

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

### Option 2: Professional Flask Interface
1. Follow steps 1-3 above for setup
2. Run the Flask application:
   ```bash
   python flask_app/app.py
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
