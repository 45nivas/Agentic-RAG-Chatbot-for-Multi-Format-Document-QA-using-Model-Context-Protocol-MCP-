import streamlit as st
import os
import tempfile
import traceback

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 Simple RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)

if prompt := st.chat_input("Ask about your documents"):
    if not uploaded_files:
        st.error("Please upload documents first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            with st.spinner("Processing..."):
                import sys
                sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                
                temp_dir = tempfile.mkdtemp()
                file_paths = []
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    file_paths.append(file_path)
                
                all_text = ""
                for file_path in file_paths:
                    try:
                        if file_path.endswith('.txt'):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                all_text += f.read() + "\n"
                        elif file_path.endswith('.pdf'):
                            try:
                                from PyPDF2 import PdfReader
                                reader = PdfReader(file_path)
                                for page in reader.pages:
                                    all_text += page.extract_text() + "\n"
                            except:
                                all_text += f"Could not read PDF: {file_path}\n"
                    except:
                        continue
                
                context = all_text[:2000]
                
                try:
                    import os
                    import google.generativeai as genai
                    from dotenv import load_dotenv
                    
                    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
                    api_key = os.getenv("GEMINI_API_KEY")
                    
                    if not api_key:
                        st.error("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
                        st.stop()
                    
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    
                    full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
                    response = model.generate_content(full_prompt)
                    answer = response.text if hasattr(response, 'text') else "Could not generate response"
                    
                except Exception as e:
                    answer = f"Error with Gemini: {str(e)}"
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.code(traceback.format_exc())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
