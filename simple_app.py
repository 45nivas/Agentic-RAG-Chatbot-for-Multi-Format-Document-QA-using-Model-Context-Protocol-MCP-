from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-for-demo')

@app.route('/')
def index():
    return """
    <html>
    <head><title>RAG Chatbot - Coming Soon</title></head>
    <body style="font-family: Arial; text-align: center; padding: 50px;">
        <h1>🤖 Agentic RAG Chatbot</h1>
        <h2>Multi-Format Document QA System</h2>
        <p>This is a demonstration deployment of my RAG chatbot portfolio project.</p>
        <p><strong>Features:</strong></p>
        <ul style="text-align: left; max-width: 500px; margin: 0 auto;">
            <li>Multi-Agent RAG Architecture</li>
            <li>Google Gemini AI Integration</li>
            <li>ChromaDB Vector Database</li>
            <li>PDF, DOCX, PPTX Processing</li>
            <li>Professional Flask Web Interface</li>
        </ul>
        <p><em>Full AI features will be enabled once deployment stabilizes.</em></p>
        <br>
        <p>🔗 <strong>GitHub:</strong> <a href="https://github.com/45nivas/Agentic-RAG-Chatbot-for-Multi-Format-Document-QA-using-Model-Context-Protocol-MCP-">View Source Code</a></p>
    </body>
    </html>
    """

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'Agentic RAG Chatbot',
        'version': '1.0.0'
    })

@app.route('/api/test')
def test_api():
    return jsonify({
        'message': 'RAG Chatbot API is working!',
        'gemini_key_configured': bool(os.environ.get('GEMINI_API_KEY')),
        'status': 'ready_for_ai_features'
    })

if __name__ == '__main__':
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host=host, port=port)
