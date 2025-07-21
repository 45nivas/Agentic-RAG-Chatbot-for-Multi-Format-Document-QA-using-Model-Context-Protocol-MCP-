# Smart Document Q&A Bot

An intelligent chatbot that can answer questions about your documents using AI. Upload PDFs, Word docs, PowerPoints and more - then ask questions and get instant answers!

## What it does
- Upload multiple document types (PDF, DOCX, PPTX, CSV, TXT, Markdown)  
- Ask questions in natural language
- Get AI-powered answers with source references
- Multi-turn conversations that remember context
- Clean web interface built with Streamlit

## Tech Stack
- **Frontend**: Streamlit web app
- **AI Engine**: Google Gemini 1.5 Flash
- **Vector Search**: ChromaDB with sentence transformers
- **Document Processing**: PyPDF2, python-docx, python-pptx
- **Architecture**: Multi-agent system with custom message passing

## Quick Start
1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run ui/app.py
   ```
4. Open your browser to `http://localhost:8501`
5. Upload documents and start asking questions!

## How it works
The system uses a multi-agent architecture where different components handle specific tasks:

- **Document Parser**: Extracts text from various file formats
- **Content Chunker**: Breaks documents into searchable segments  
- **Vector Search**: Finds relevant content using semantic similarity
- **AI Response**: Generates natural language answers using Google Gemini

## Project Structure
```
├── agents/          # Core processing logic
│   ├── coordinator_agent.py    # Main workflow coordinator
│   ├── ingestion_agent.py      # Document parsing
│   ├── retrieval_agent.py      # Vector search
│   └── llm_response_agent.py   # AI response generation
├── ui/              # Web interface
│   └── app.py       # Main Streamlit application
└── requirements.txt # Dependencies
```

## Sample Workflow (Message Passing with MCP)

**Example scenario:**
```
User uploads: sales_review.pdf, metrics.csv
User: "What KPIs were tracked in Q1?"
```

**Message flow:**
```
➡️ UI forwards to CoordinatorAgent
➡️ Coordinator triggers:
   🔸 IngestionAgent → parses documents
   🔸 RetrievalAgent → finds relevant chunks  
   🔸 LLMResponseAgent → formats prompt & calls LLM
➡️ Chatbot shows answer + source chunks
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

## Usage Example
1. Upload a PDF of your company's annual report
2. Ask: "What were the key financial highlights this year?"
3. Get an AI-generated answer with specific references to relevant sections
4. Follow up with: "What about the challenges mentioned?"
5. The bot remembers the context and provides relevant answers

## Notes
- Make sure you have a stable internet connection for the AI API calls
- Larger documents may take a moment to process initially  
- The system works best with text-heavy documents (PDFs with images may have limited text extraction)

---

We're looking forward to seeing your creativity and problem-solving in action. Good luck and happy coding!
