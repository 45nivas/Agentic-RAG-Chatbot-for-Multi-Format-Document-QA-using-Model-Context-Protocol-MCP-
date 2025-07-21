"""
LLM utilities for Google Gemini API integration.
"""
from typing import List
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyCgRmst1xIMP_N9EJyQjrSwSIYRn_kDcWs"

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
