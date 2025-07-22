import streamlit as st
import sys
import os
import tempfile
import io
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

# Setup page
st.set_page_config(page_title="🤖 RAG Document Chatbot", layout="wide")
st.title("🤖 Smart Document Q&A Bot")
st.markdown("Upload documents and ask questions about them!")

# Simple MCP Message class (no external dependencies)
@dataclass
class MCPMessage:
    sender: str
    receiver: str
    type: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    payload: Dict[str, Any] = field(default_factory=dict)

def extract_text_simple(uploaded_files):
    """Extract text from uploaded files - FAST method"""
    all_chunks = []
    for file in uploaded_files:
        try:
            file_content = ""
            if file.name.endswith(('.txt', '.md')):
                file_content = str(file.read(), "utf-8")
            elif file.name.endswith('.pdf'):
                try:
                    from PyPDF2 import PdfReader
                    pdf_reader = PdfReader(io.BytesIO(file.read()))
                    for page in pdf_reader.pages:
                        file_content += page.extract_text() + "\n"
                except:
                    file_content = "[Could not read PDF]"
            else:
                file_content = f"[{file.name} - File type not supported]"
            
            # Simple chunking - split by paragraphs
            chunks = [chunk.strip() for chunk in file_content.split('\n\n') if chunk.strip()]
            all_chunks.extend(chunks[:10])  # Limit to 10 chunks per file for speed
        except:
            continue
    
    return all_chunks

def simple_retrieval(chunks, query, top_k=3):
    """Simple keyword-based retrieval - NO vector embeddings needed"""
    query_words = query.lower().split()
    scored_chunks = []
    
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for word in query_words if word in chunk_lower)
        if score > 0:
            scored_chunks.append((score, chunk))
    
    # Sort by score and return top chunks
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored_chunks[:top_k]]

def process_with_agents(uploaded_files, prompt):
    """Fast multi-agent processing - NO heavy dependencies"""
    trace_id = str(uuid.uuid4())[:8]
    
    try:
        # Step 1: Ingestion Agent (FAST)
        chunks = extract_text_simple(uploaded_files)
        ingest_msg = MCPMessage(
            sender="IngestionAgent",
            receiver="RetrievalAgent", 
            type="CHUNKIFY_RESULT",
            trace_id=trace_id,
            payload={"chunks": chunks, "file_count": len(uploaded_files)}
        )
        
        # Step 2: Retrieval Agent (FAST - no embeddings)
        relevant_chunks = simple_retrieval(chunks, prompt)
        retrieval_msg = MCPMessage(
            sender="RetrievalAgent",
            receiver="LLMResponseAgent",
            type="RETRIEVAL_RESULT", 
            trace_id=trace_id,
            payload={"retrieved_context": relevant_chunks, "query": prompt}
        )
        
        # Step 3: LLM Response Agent
        if relevant_chunks:
            context = "\n\n".join(relevant_chunks[:3])
            
            # Call Gemini API
            import google.generativeai as genai
            from dotenv import load_dotenv
            
            load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
            api_key = os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                answer = "❌ GEMINI_API_KEY not found. Check your .env file."
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                full_prompt = f"""Based on the document content below, answer the question clearly.

Document Content:
{context}

Question: {prompt}

Answer:"""
                
                try:
                    response = model.generate_content(full_prompt)
                    answer = response.text if hasattr(response, 'text') and response.text else "Could not generate response"
                except Exception as e:
                    if "rate limit" in str(e).lower() or "429" in str(e):
                        answer = "🚫 Rate limit reached! Please wait a few minutes."
                    else:
                        answer = f"❌ API Error: {str(e)[:100]}..."
        else:
            answer = "No relevant content found in the uploaded documents."
        
        llm_msg = MCPMessage(
            sender="LLMResponseAgent",
            receiver="UI",
            type="RESPONSE",
            trace_id=trace_id,
            payload={"answer": answer, "query": prompt}
        )
        
        return {
            "answer": f"🤖 **Multi-Agent Response:**\n\n{answer}\n\n🔍 *Found {len(relevant_chunks)} relevant sections*",
            "source_context": relevant_chunks,
            "trace_id": trace_id,
            "messages": [ingest_msg, retrieval_msg, llm_msg]
        }
        
    except Exception as e:
        return {
            "answer": f"❌ Error: {str(e)[:100]}...",
            "source_context": [],
            "trace_id": trace_id
        }

def process_simple(uploaded_files, prompt):
    """Simple mode - even faster"""
    try:
        chunks = extract_text_simple(uploaded_files)
        context = "\n\n".join(chunks[:5])[:3000]  # Limit context
        
        import google.generativeai as genai
        from dotenv import load_dotenv
        
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            return "❌ GEMINI_API_KEY not found. Check your .env file."
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        full_prompt = f"""Based on the document content below, answer the question clearly.

Document Content:
{context}

Question: {prompt}

Answer:"""
        
        response = model.generate_content(full_prompt)
        answer = response.text if hasattr(response, 'text') and response.text else "Could not generate response"
        
        return f"⚡ **Simple Mode Response:**\n\n{answer}\n\n📊 *Processed {len(uploaded_files)} file(s)*"
        
    except Exception as e:
        if "rate limit" in str(e).lower() or "429" in str(e):
            return "🚫 Rate limit reached! Please wait a few minutes."
        else:
            return f"❌ Error: {str(e)[:100]}..."

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# File upload
st.subheader("📁 Upload Documents")
uploaded_files = st.file_uploader(
    "Choose files", 
    accept_multiple_files=True,
    type=['txt', 'pdf', 'docx', 'pptx', 'csv', 'md'],
    help="Upload PDF, DOCX, PPTX, CSV, TXT, or MD files"
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded!")
    file_names = [f.name for f in uploaded_files]
    st.write("📄 Files: " + ", ".join(file_names))

# Settings
col1, col2 = st.columns([1, 1])
with col1:
    use_agents = st.checkbox("🤖 Use Multi-Agent System", value=True, 
                            help="Advanced processing with multiple agents")
with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.divider()

# Chat section
st.subheader("💬 Chat")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Show source context if available in message history
        if message["role"] == "assistant" and "source_context" in message:
            source_context = message["source_context"]
            trace_id = message.get("trace_id", "unknown")
            if source_context:
                with st.expander("📋 View Source Context", expanded=False):
                    st.write(f"**Trace ID:** `{trace_id}`")
                    for i, context in enumerate(source_context):
                        st.write(f"**📄 Source {i+1}:**")
                        st.write(f"> {context[:200]}{'...' if len(context) > 200 else ''}")
                        if i < len(source_context) - 1:
                            st.divider()

# Chat input
prompt = st.chat_input("Ask about your documents...")

# Process the input
if prompt:
    if not uploaded_files:
        st.error("Please upload documents first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process and show response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    if use_agents:
                        result = process_with_agents(uploaded_files, prompt)
                        answer = result["answer"]
                        source_context = result.get("source_context", [])
                        trace_id = result.get("trace_id", "unknown")
                        
                        st.write(answer)
                        
                        # Show source context if available
                        if source_context:
                            with st.expander("📋 View Source Context", expanded=False):
                                st.write(f"**Trace ID:** `{trace_id}`")
                                for i, context in enumerate(source_context):
                                    st.write(f"**📄 Source {i+1}:**")
                                    st.write(f"> {context[:200]}{'...' if len(context) > 200 else ''}")
                                    if i < len(source_context) - 1:
                                        st.divider()
                        
                        # Store message with source context for history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "source_context": source_context,
                            "trace_id": trace_id
                        })
                    else:
                        # Simple mode
                        answer = process_simple(uploaded_files, prompt)
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"❌ Unexpected error: {str(e)[:100]}..."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Force rerun to update the UI
        st.rerun()
