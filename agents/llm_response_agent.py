import os
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv
import time
from .mcp import MCPMessage

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

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
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.gemini.generate_content(prompt)
                if hasattr(response, "text") and response.text:
                    return response.text.strip()
                else:
                    return f"Error: No response text from Gemini. Response: {str(response)}"
            except Exception as e:
                error_str = str(e)
                if "429" in error_str and "quota" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = 30 + (attempt * 10)
                        return f"⏱️ Rate limit reached. Please wait {wait_time} seconds and try again. (Free tier: 15 requests/minute)"
                    else:
                        return f"❌ Rate limit exceeded. Please wait a few minutes before trying again, or consider upgrading your Gemini API plan."
                else:
                    return f"Error calling Gemini API: {error_str}"


class LLMResponseAgent:
    def __init__(self):
        self.client = LLMClient()
    
    def generate_response(self, context: List[str], query: str) -> MCPMessage:
        response_text = self.client.ask(context, query)
        return MCPMessage(
            sender="LLMResponseAgent",
            receiver="UI",
            type="RESPONSE",
            payload={"answer": response_text, "query": query}
        )
