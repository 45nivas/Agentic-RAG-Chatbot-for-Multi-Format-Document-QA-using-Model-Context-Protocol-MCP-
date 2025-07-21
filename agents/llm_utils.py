"""
LLM utilities for Google Gemini API integration.
"""
import os
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

class LLMClient:
    def __init__(self, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = model
        self.gemini = genai.GenerativeModel(model)

    def ask(self, context: List[str], query: str) -> str:
        prompt = f"Context:\n{chr(10).join(context)}\n\nQuestion: {query}\nAnswer:"
        try:
            response = self.gemini.generate_content(prompt)
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            else:
                return f"Error: No response text from Gemini. Response: {str(response)}"
        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"
