# Multi-Agent RAG Chatbot

[![Live Demo](https://img.shields.io/badge/🌐%20Live%20Demo-Click%20Here-blue?style=for-the-badge)](https://agentic-rag-chatbot-for-multi-format-jwlk.onrender.com/)

## 🚀 Smart Document Q&A Bot

Professional RAG system with dual deployment strategy for development and production.

## ✨ Key Features

- **6 document formats** - PDF, DOCX, PPTX, CSV, TXT, MD
- **Voice input** with microphone 🎤  
- **Dual Architecture** - Advanced for dev, optimized for production
- **Multi-agent system** - 4 specialized AI agents
- **Real-time chat** interface
- **Sub-3 second responses**
- **Professional UI** with modern design

## 🏗️ Triple Architecture

### **Efficient Advanced** (`flask_app/app_efficient.py`) - **RECOMMENDED** ⭐
- **LangChain Integration** - Modern AI framework (150 lines only!)
- **HuggingFace Embeddings** - Efficient sentence transformers
- **ChromaDB** - Persistent vector storage
- **Smart Processing** - Automatic chunking & embedding

### **Full Advanced** (`flask_app/app.py`)
- **Sentence Transformers** - Complete implementation (800 lines)
- **ChromaDB** - Full featured with HNSW indexing
- **Multi-Agent MCP** - Enterprise architecture
- **Advanced Processing** - Comprehensive document handling

### **Production Deployment** (`app.py`)
- **Optimized TF-IDF** - Enhanced with bigrams and better parameters
- **Lightweight** - Fits in 512MB Render memory limit  
- **Fast Startup** - No heavy model downloads
- **Professional UI** - Same modern interface

## 🚀 Quick Start

### **Efficient Advanced (RECOMMENDED)** ⭐
```bash
# Modern LangChain approach - Only 150 lines!
cd flask_app
pip install -r requirements-efficient.txt
echo "GEMINI_API_KEY=your_key" > ../.env
python app_efficient.py
```

### **Full Advanced Development**
```bash
# Complete implementation - 800 lines with all features
cd flask_app
pip install -r requirements-dev.txt
echo "GEMINI_API_KEY=your_key" > ../.env
python app.py
```

### **Production Deployment (Render)**
```bash
# Lightweight version (uses requirements.txt automatically)
echo "GEMINI_API_KEY=your_key" > .env
python app.py
```

## 📋 Requirements Files

| File | Purpose | AI Features | Memory | Deployment |
|------|---------|-------------|---------|------------|
| `requirements.txt` | **Production** | Enhanced TF-IDF | ~50MB | ✅ Render Ready |
| `requirements-local.txt` | **Local Full AI** | Sentence Transformers + ChromaDB | ~300MB | 💻 Development |
| `flask_app/requirements-dev.txt` | **Advanced Dev** | Complete AI Suite + Tools | ~500MB | 🔬 Research |

## 💡 How It Works

1. **Upload** documents using drag & drop
2. **Processing** - AI agents parse and chunk content  
3. **Ask questions** - Type or use voice input 🎤
4. **Get answers** - AI provides context-aware responses

## 🏗️ Architecture

```
Upload → Document Agents → Vector Search → AI Response
   ↓          ↓              ↓             ↓
Files → Parse/Chunk → Find Context → Generate Answer
```

## 🛠️ Tech Stack

### **Development Version**
- **AI**: Sentence Transformers (all-MiniLM-L6-v2) + Google Gemini 1.5 Flash
- **Database**: ChromaDB with persistent storage and HNSW indexing
- **Embeddings**: 384-dimensional semantic vectors
- **ML**: PyTorch, Transformers, SentenceTransformers

### **Production Version**
- **AI**: Enhanced TF-IDF + Google Gemini 1.5 Flash
- **Search**: Optimized TF-IDF with bigrams and smart parameters
- **Memory**: Lightweight, under 100MB total
- **Deployment**: Cloud-optimized for Render/Heroku

### **Common Features**
- **Backend**: Flask with multi-agent architecture
- **Frontend**: HTML/CSS/JS with Web Speech API
- **Documents**: PyPDF2, python-docx, python-pptx
- **Agents**: Coordinator, Ingestion, Retrieval, LLM

## 📁 Project Structure

```
├── app.py                     # Production deployment (lightweight)
├── requirements.txt           # Production dependencies (Render)
├── requirements-local.txt     # Local development with AI
├── flask_app/
│   ├── app.py                # Advanced development version
│   └── requirements-dev.txt  # Complete AI development suite
├── agents/                    # Multi-agent system
├── static/                    # Frontend assets
├── templates/                 # HTML templates
└── .env                      # API configuration
```

## 🎯 Professional Features

- ✅ **Dual deployment strategy** - Optimized for both development and production
- ✅ **Modern AI stack** - Sentence Transformers with ChromaDB persistence
- ✅ **Multi-agent architecture** - Enterprise-level AI coordination
- ✅ **Voice input integration** - Hands-free document interaction
- ✅ **Professional UI/UX** - Modern, responsive design
- ✅ **Smart chunking** - Intelligent document processing with context
- ✅ **Session management** - Persistent conversation history
- ✅ **Production ready** - Cloud deployment optimized

## 🌐 Deployment Options

### **Local Development**
```bash
# Full AI features with Sentence Transformers
cd flask_app
pip install -r requirements-dev.txt
python app.py
# → http://localhost:5000
```

### **Local with Basic AI**
```bash
# Sentence Transformers without heavy dev tools
pip install -r requirements-local.txt
python app.py
# → http://localhost:5000
```

### **Production Cloud**
```bash
# Lightweight TF-IDF for cloud deployment
git push origin main
# → Automatic Render deployment
```

## 🔧 Environment Setup

### **Development**
- Python 3.8+
- 4GB+ RAM (for Sentence Transformers)
- GEMINI_API_KEY

### **Production** 
- Python 3.8+
- 512MB RAM (cloud compatible)
- GEMINI_API_KEY

---
**Built with professional dual-architecture for optimal AI performance** 🚀
