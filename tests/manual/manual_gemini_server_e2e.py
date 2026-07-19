import os
import sys
import json
import time
import subprocess
import requests

# Reconfigure stdout/stderr to handle UTF-8 printing safely on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(os.getcwd())

def run_server_e2e():
    print("==================================================")
    print("STARTING PERSISTENT FLASK SERVER IN BACKGROUND")
    print("==================================================")
    
    # Start app.py in background
    # Set FLASK_ENV=production to match production settings
    env = os.environ.copy()
    env["FLASK_ENV"] = "production"
    env["DISABLE_GEMINI"] = "false"
    if "OLLAMA_MODEL" in env:
        del env["OLLAMA_MODEL"]
        
    server_process = subprocess.Popen(
        [sys.executable, "app.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1
    )
    
    print("Waiting for server to initialize and load embedding models...")
    # Monitor stdout until server is initialized
    initialized = False
    start_time = time.time()
    while time.time() - start_time < 240: # Give it up to 240 seconds (4 minutes)
        line = server_process.stdout.readline()
        if not line:
            break
        print(f"[Server Log] {line.strip()}")
        if "Advanced RAG Agent System initialized" in line or "Running on http" in line or "Debugger PIN" in line:
            initialized = True
            print("\n✅ Server initialized successfully!")
            break
            
    if not initialized:
        print("Error: Server failed to initialize in time.")
        server_process.terminate()
        return

    # Wait a couple of seconds to ensure socket is open
    time.sleep(3)
    
    session = requests.Session()
    base_url = "http://127.0.0.1:5000"
    pdf_path = r"c:\Users\matta\OneDrive\Desktop\resume_projects\Agentic-RAG-Chatbot-for-Multi-Format-Document-QA-using-Model-Context-Protocol-MCP--main\MR. N SATYENDRA.pdf"
    
    try:
        print("\n==================================================")
        print("STEP 1: UPLOADING LAB REPORT TO /api/upload")
        print("==================================================")
        
        upload_start = time.time()
        with open(pdf_path, 'rb') as f:
            files = {'files': ('MR. N SATYENDRA.pdf', f, 'application/pdf')}
            response = session.post(f"{base_url}/api/upload", files=files)
        upload_end = time.time()
        
        print(f"Upload Response Status: {response.status_code}")
        print(f"Upload and extraction completed in {upload_end - upload_start:.2f} seconds.")
        
        if response.status_code != 200:
            print(f"Upload failed: {response.text}")
            return
            
        res_json = response.json()
        profile = res_json.get('profile', {})
        extraction_incomplete = res_json.get('extraction_incomplete', False)
        extraction_error = res_json.get('extraction_error', None)
        
        print("\n--- Demographics Extracted ---")
        demographics = profile.get('demographics', {})
        print(json.dumps(demographics, indent=2))
        print(f"Name check: '{demographics.get('name')}'")
        
        print(f"\nExtraction Incomplete: {extraction_incomplete}")
        print(f"Extraction Error/Warning: {extraction_error}")
        
        print("\n--- Biomarkers Extracted ---")
        biomarkers = profile.get('biomarkers', [])
        print(json.dumps(biomarkers[:5], indent=2)) # Print first 5 biomarkers
        print(f"Total biomarkers extracted: {len(biomarkers)}")

        # Wait 10 seconds before chat to prevent API rate limiting
        print("\nSleeping 10 seconds to cool down rate limits...")
        time.sleep(10)

        print("\n==================================================")
        print("STEP 2: SENDING CHAT QUERY TO /api/chat")
        print("==================================================")
        
        chat_payload = {
            "message": "What does my blood report say and what should I do?"
        }
        
        chat_start = time.time()
        chat_response = session.post(f"{base_url}/api/chat", json=chat_payload)
        chat_end = time.time()
        
        print(f"Chat Response Status: {chat_response.status_code}")
        print(f"Chat execution completed in {chat_end - chat_start:.2f} seconds.")
        
        if chat_response.status_code != 200:
            print(f"Chat failed: {chat_response.text}")
            return
            
        chat_json = chat_response.json()
        
        print("\n--- Plan and Critique Outputs Check ---")
        meal_plan = chat_json.get('meal_plan')
        training_plan = chat_json.get('training_plan')
        audit_report = chat_json.get('audit_report')
        bio_age_results = chat_json.get('bio_age_results')
        critique = chat_json.get('critique')
        mcp_trace = chat_json.get('mcp_trace', [])
        
        print(f"Meal Plan Type: {type(meal_plan).__name__}")
        print(f"Training Plan Type: {type(training_plan).__name__}")
        print(f"Audit Report Present: {audit_report is not None}")
        print(f"Bio Age Results Present: {bio_age_results is not None}")
        print(f"Critique Present: {critique is not None}")
        
        print("\n--- MCP Trace Execution Timings ---")
        for trace in mcp_trace:
            print(f"Agent: {trace.get('receiver')} | Type: {trace.get('type')} | Timestamp: {trace.get('timestamp') or 'N/A'}")

        print("\n==================================================")
        print("STEP 3: DOWNLOAD COMPILED PDF REPORT")
        print("==================================================")
        
        pdf_response = session.get(f"{base_url}/api/report/download")
        print(f"PDF Download Status: {pdf_response.status_code}")
        if pdf_response.status_code == 200:
            pdf_bytes = pdf_response.content
            print(f"PDF successfully generated. File size: {len(pdf_bytes)} bytes.")
            save_path = os.path.join(os.path.dirname(pdf_path), "NutriMind_E2E_RealGemini_Report.pdf")
            with open(save_path, "wb") as pdf_file:
                pdf_file.write(pdf_bytes)
            print(f"PDF saved to: {save_path}")
        else:
            print(f"PDF generation failed: {pdf_response.text}")
            
    finally:
        print("\nShutting down Flask server...")
        server_process.terminate()
        server_process.wait()
        print("Flask server terminated.")

if __name__ == "__main__":
    run_server_e2e()
