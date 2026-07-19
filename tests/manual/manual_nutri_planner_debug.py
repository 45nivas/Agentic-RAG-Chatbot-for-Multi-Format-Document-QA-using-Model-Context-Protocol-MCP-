import os
import sys
from google import genai
from google.genai import types

sys.path.append(os.getcwd())

from dotenv import load_dotenv
env_path = os.path.join(os.getcwd(), '.env')
load_dotenv(env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def test_truncation(max_tokens, response_mime_type=None):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # A prompt designed to produce a very detailed, verbose response (plain text mode)
    prompt = """You are a senior sports dietitian and clinical endocrinologist.
Create a highly detailed, comprehensive day of nutrition and conditioning advice for a 23-year-old male athlete.
For breakfast, lunch, dinner, and snacks:
1. Describe the exact meal and prep steps in great detail.
2. Provide a 3-paragraph clinical justification for why this meal is perfect for hormonal optimization, muscle recovery, and energy levels.
3. List all macronutrient targets.

Be extremely verbose and thorough in your descriptions."""

    config_params = {"max_output_tokens": max_tokens}
    if response_mime_type:
        config_params["response_mime_type"] = response_mime_type
        
    config = types.GenerateContentConfig(**config_params)
    
    mime_str = response_mime_type or "OMITTED (Plain Text)"
    print(f"Testing config: max_output_tokens={max_tokens}, response_mime_type={mime_str}")
    
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=config
        )
        print("Finish Reason:", response.candidates[0].finish_reason)
        print("Output length:", len(response.text))
        print("Output snippet (first 150 chars):")
        print(response.text[:150])
        print("Output snippet (last 150 chars):")
        print(response.text[-150:])
        print("="*50)
    except Exception as e:
        print(f"Error: {e}")
        print("="*50)

if __name__ == "__main__":
    # Test 1: max_output_tokens=4096, response_mime_type OMITTED
    test_truncation(4096, response_mime_type=None)
    # Test 2: max_output_tokens=2048, response_mime_type OMITTED
    test_truncation(2048, response_mime_type=None)
