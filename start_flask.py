#!/usr/bin/env python
"""
Startup script for the Flask RAG Chatbot
"""
import os
import sys
import subprocess

def start_flask_app():
    """Start the Flask application"""
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    flask_dir = os.path.join(script_dir, 'flask_app')
    
    # Set the working directory
    os.chdir(flask_dir)
    
    # Start the Flask app
    subprocess.run([sys.executable, 'app.py'])

if __name__ == "__main__":
    start_flask_app()
