import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agents.coordinator_agent import CoordinatorAgent

st.set_page_config(page_title="Document QA Chatbot", layout="wide")
st.title("📄 Smart Document Q&A Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

uploaded_files = st.file_uploader("📎 Upload documents (PDF, DOCX, PPTX, CSV, TXT, MD)", accept_multiple_files=True)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
    st.session_state.uploaded_files = uploaded_files

st.subheader("💬 Chat with your documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.uploaded_files:
        st.error("Please upload documents first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing your question..."):
                try:
                    file_paths = []
                    for file in st.session_state.uploaded_files:
                        temp_path = f"temp_{file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                        file_paths.append(temp_path)
                    
                    st.write(f"📁 Processing {len(file_paths)} files: {[f.split('_', 1)[1] for f in file_paths]}")
                    
                    coordinator = CoordinatorAgent()
                    messages = coordinator.process(file_paths, prompt)
                    
                    st.write(f"🔄 Got {len(messages)} messages from agents")
                    
                    if len(messages) >= 3:
                        ingest_msg = messages[0]
                        retrieval_msg = messages[1] 
                        llm_msg = messages[2]
                        
                        # Debug: Show chunks found
                        chunks_found = len(ingest_msg.payload.get("chunks", []))
                        st.write(f"📄 Extracted {chunks_found} text chunks")
                        
                        context_found = len(retrieval_msg.payload.get("retrieved_context", []))
                        st.write(f"🔍 Found {context_found} relevant sections")
                        
                        if llm_msg.payload.get("answer"):
                            answer = llm_msg.payload["answer"]
                            
                            st.markdown("### 💬 Answer:")
                            st.write(answer)
                            
                            if retrieval_msg.payload.get("retrieved_context"):
                                with st.expander("📋 Source Context"):
                                    for i, context in enumerate(retrieval_msg.payload["retrieved_context"]):
                                        st.write(f"**Chunk {i+1}:** {context[:200]}...")
                            
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        else:
                            error_msg = "Sorry, I couldn't generate an answer. Please try again."
                            st.write(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    else:
                        error_msg = "Not enough processing steps completed. Please try again."
                        st.write(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
                    for path in file_paths:
                        if os.path.exists(path):
                            os.remove(path)
                            
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
