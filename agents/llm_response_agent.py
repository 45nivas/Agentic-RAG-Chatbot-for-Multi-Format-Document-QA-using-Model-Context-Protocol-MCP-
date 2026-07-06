import os
import re
import time
import random
import logging
import json
import requests
import threading
from typing import List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv
from .mcp import MCPMessage

logger = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

# Open-Source Fallback System Tiers & Concurrency Locks
_local_pipeline = None
from .locks import _local_pipeline_lock

_ollama_lock = threading.Lock()
_ollama_checked = False
_ollama_available_models = None

def get_installed_ollama_models() -> List[str]:
    global _ollama_available_models, _ollama_checked
    with _ollama_lock:
        if _ollama_checked:
            return _ollama_available_models or []
            
        _ollama_checked = True
        try:
            logger.info("🔍 Checking installed local Ollama models...")
            r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
            if r.status_code == 200:
                res = r.json()
                models = [m["name"] for m in res.get("models", [])]
                _ollama_available_models = models
                logger.info(f"✅ Ollama is running locally. Installed models: {models}")
                return models
        except Exception:
            logger.info("ℹ️ Ollama is not running locally.")
            _ollama_available_models = []
            
        return []

def call_ollama(prompt: str) -> str:
    installed_models = get_installed_ollama_models()
    if not installed_models:
        return ""
        
    # Check if any of our preferred models are installed
    preferred = ["qwen2.5:latest", "llama3:latest", "mistral:latest", "phi3:latest", "gemma2:latest", "llama3", "mistral"]
    target_model = None
    for model in preferred:
        if model in installed_models or any(m.startswith(model.split(':')[0]) for m in installed_models):
            target_model = model
            break
            
    if not target_model and installed_models:
        # Fallback to the first available model
        target_model = installed_models[0]
        
    if not target_model:
        return ""
        
    url = "http://localhost:11434/api/generate"
    try:
        logger.info(f"🤖 Querying local Ollama model '{target_model}'...")
        response = requests.post(
            url,
            json={
                "model": target_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2
                }
            },
            timeout=5.0
        )
        if response.status_code == 200:
            resp_json = response.json()
            text = resp_json.get("response", "").strip()
            if text:
                logger.info(f"Ollama '{target_model}' response successful!")
                return text
    except Exception as e:
        logger.warning(f"Ollama call failed for '{target_model}': {e}")
    return ""

def call_huggingface_api(prompt: str) -> str:
    # Try to find a token in env
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACE_API_TOKEN")
    if not hf_token:
        logger.info("ℹ️ No HuggingFace API token found in environment. Skipping Serverless Inference API to prevent timeouts.")
        return ""
        
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Try a couple of excellent open source instruction models. 
    # Qwen models are highly public and don't require gate agreements.
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3"
    ]
    
    for model_id in models:
        try:
            url = f"https://api-inference.huggingface.co/models/{model_id}"
            logger.info(f"Trying HuggingFace Serverless Inference API with model '{model_id}'...")
            
            # Standard Hugging Face format
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1024,
                    "temperature": 0.2,
                    "return_full_text": False
                }
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=8.0)
            if response.status_code == 200:
                resp_json = response.json()
                if isinstance(resp_json, list) and len(resp_json) > 0:
                    text = resp_json[0].get("generated_text", "").strip()
                    if text:
                        logger.info(f"HuggingFace API model '{model_id}' response successful!")
                        return text
                elif isinstance(resp_json, dict):
                    text = resp_json.get("generated_text", "").strip()
                    if text:
                        logger.info(f"HuggingFace API model '{model_id}' response successful!")
                        return text
            elif response.status_code == 503:
                # Model is loading
                logger.info(f"HF model '{model_id}' is currently loading. Trying next...")
        except Exception as e:
            logger.debug(f"HF Inference error for '{model_id}': {e}")
    return ""

def call_local_transformers(prompt: str) -> str:
    global _local_pipeline
    try:
        from transformers import pipeline
        import torch
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        
        def run_inference():
            global _local_pipeline
            with _local_pipeline_lock:
                if _local_pipeline is None:
                    logger.info("Initializing ultra-lightweight local open-source model (Qwen/Qwen2.5-0.5B-Instruct) on CPU...")
                    # We remove device_map="auto" to run perfectly on CPU out-of-the-box without requiring `accelerate`
                    _local_pipeline = pipeline(
                        "text-generation",
                        model="Qwen/Qwen2.5-0.5B-Instruct",
                        torch_dtype=torch.float32
                    )
            
                # RUN THE INFERENCE INSIDE THE LOCK to prevent parallel CPU thrashing!
                logger.info("Running local open-source model inference on CPU (serialized)...")
                messages = [
                    {"role": "user", "content": prompt}
                ]
                
                outputs = _local_pipeline(
                    messages,
                    max_new_tokens=512,
                    temperature=0.2,
                    do_sample=True
                )
                
                if outputs and isinstance(outputs, list):
                    gen_text = outputs[0].get("generated_text", "")
                    if isinstance(gen_text, list):
                        for item in reversed(gen_text):
                            if item.get("role") == "assistant":
                                return item.get("content", "").strip()
                    elif isinstance(gen_text, str):
                        result = gen_text
                        if result.startswith(prompt):
                            result = result[len(prompt):]
                        return result.strip()
                return ""

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(run_inference)
        try:
            res_text = future.result(timeout=30)
            executor.shutdown(wait=False)
            return res_text
        except TimeoutError:
            logger.error("❌ Local CPU inference exceeded 30s timeout")
            executor.shutdown(wait=False)
            raise TimeoutError("Local CPU inference exceeded 30s timeout")
        except Exception as e:
            executor.shutdown(wait=False)
            raise e
    except Exception as e:
        logger.error(f"Failed local transformers inference: {e}")
    return ""

def call_opensource_fallback(prompt: str) -> str:
    logger.info("🔄 Initiating 3-Tier Open-Source Fallback System...")
    
    # Tier 1: Local Ollama Server (super fast, fully offline, user-controlled)
    try:
        response = call_ollama(prompt)
        if response:
            return response
    except Exception as e:
        logger.debug(f"Ollama tier failed: {e}")
        
    # Tier 2: Hugging Face Free Serverless Inference API (extremely high accuracy, fast remote)
    try:
        response = call_huggingface_api(prompt)
        if response:
            return response
    except Exception as e:
        logger.debug(f"Hugging Face API tier failed: {e}")
        
    # Tier 3: Local CPU Transformers (100% private, completely offline, zero dependencies on external APIs)
    try:
        response = call_local_transformers(prompt)
        if response:
            return response
    except Exception as e:
        logger.debug(f"Local CPU Transformers tier failed: {e}")
        
    return ""

# Resiliency Decorator: Exponential Backoff + Jitter Retry Handler
def resilient_llm_call(
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_factor: float = 2.0,
    max_retries: int = 4,
    jitter: bool = True
):
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e)
                    
                    # If it's a quota exceeded / rate limit / 429 error, don't wait and retry!
                    # Fall back immediately to save user's time and prevent HTTP timeouts.
                    is_quota_error = "429" in error_str or "quota" in error_str.lower() or "limit" in error_str.lower() or "resourceexhausted" in error_str.lower()
                    
                    exc_name = type(e).__name__.lower()
                    is_timeout_error = "timeout" in exc_name or "deadline" in exc_name or "timedout" in exc_name or "deadlineexceeded" in exc_name or "deadline exceeded" in error_str.lower()
                    
                    if is_quota_error or is_timeout_error:
                        reason = "Quota Exceeded / Rate Limited (429)" if is_quota_error else "Request Timeout (DeadlineExceeded)"
                        logger.warning(
                            f"⚠️ Gemini API {reason}. "
                            f"Skipping retries and falling back immediately."
                        )
                        break
                    
                    # Backoff logic
                    if attempt < max_retries - 1:
                        sleep_time = delay
                        if jitter:
                            sleep_time += random.uniform(0.1, 0.9)
                        logger.warning(
                            f"⚠️ Gemini API failure (attempt {attempt + 1}/{max_retries}). "
                            f"Reason: {error_str}. Retrying in {sleep_time:.2f} seconds..."
                        )
                        time.sleep(sleep_time)
                        delay = min(delay * exponential_factor, max_delay)
                    else:
                        logger.critical(f"❌ Max retries reached for Gemini API. Call failed.")
            
            # If all attempts failed, try open-source fallback!
            try:
                logger.info("🔌 Gemini API failed all retries. Activating Open-Source fallback LLM pipeline...")
                prompt = ""
                if len(args) > 1 and isinstance(args[1], str) and func.__name__ == "generate_with_prompt":
                    prompt = args[1]
                elif len(args) > 2 and isinstance(args[1], list) and isinstance(args[2], str):
                    context = args[1]
                    query = args[2]
                    prompt = f"Context from health documents:\n" + "\n".join(context) + f"\n\nUser query: {query}\n\nProvide a safe, detailed, and accurate clinical response based on the context."
                elif "prompt" in kwargs:
                    prompt = kwargs["prompt"]
                else:
                    str_args = [a for a in args if isinstance(a, str)]
                    if str_args:
                        prompt = str_args[-1]
                
                if prompt:
                    fallback_response = call_opensource_fallback(prompt)
                    if fallback_response:
                        logger.info("✅ Open-Source fallback LLM completed execution successfully.")
                        if func.__name__ == "generate_with_prompt":
                            return clean_and_repair_json(fallback_response)
                        return fallback_response
            except Exception as fallback_err:
                logger.error(f"❌ Open-Source fallback failed: {str(fallback_err)}")

            # Return graceful user-facing error response rather than crashing the pipeline
            return (
                f"⏱️ The clinical agent server is currently experiencing high load. "
                f"Please try again in a few seconds. (API Details: {str(last_exception)})"
            )
        return wrapper
    return decorator

def clean_and_repair_json(raw_text: str) -> str:
    """Regex JSON helper that extracts clean JSON boundaries from conversational wrappers"""
    cleaned = raw_text.strip()
    
    # 1. Strip markdown wraps: ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    
    # 2. Extract nested brackets if the LLM surrounded the JSON with text
    if not (cleaned.startswith("{") and cleaned.endswith("}")):
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace+1]
            
    return cleaned


class LLMClient:
    def __init__(self, model: str = "gemini-2.5-flash"):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = model
        self.gemini = genai.GenerativeModel(model)

    @resilient_llm_call()
    def _call_gemini_raw(self, prompt: str, generation_config: dict = None) -> str:
        response = self.gemini.generate_content(prompt, generation_config=generation_config, request_options={"timeout": 15})
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            raise RuntimeError(f"Null response payload from Gemini: {str(response)}")

    def call(self, prompt: str, system: str = None, context: List[str] = None, json_mode: bool = False) -> str:
        parts = []
        if system:
            parts.append(f"System Instructions:\n{system}\n")
        if context:
            parts.append(f"<context>\n{chr(10).join(context)}\n</context>\n")
        parts.append(prompt)
        if json_mode:
            parts.append("\nRespond only in valid JSON.")
            parts.append("Return ONLY a raw JSON object. Do not use markdown code fences. Do not include any explanation before or after the JSON.")
            
        final_prompt = "\n".join(parts)
        
        # 1. Priority 1: Try local Ollama model if running (Fast local offline LLM)
        try:
            installed_models = get_installed_ollama_models()
            if installed_models:
                logger.info("🤖 [Priority 1] Ollama is running. Attempting local model inference...")
                response_text = call_ollama(final_prompt)
                if response_text:
                    logger.info("✅ Local Ollama inference succeeded!")
                    if json_mode:
                        return clean_and_repair_json(response_text)
                    return response_text
        except Exception as e:
            logger.warning(f"⚠️ Local Ollama inference failed: {e}")
            
        # 2. Priority 2: Remote Gemini API (Fast fallback remote LLM)
        # DISABLE_GEMINI: Temporary local-testing toggle to skip Gemini API calls
        # entirely and go straight to open-source fallbacks. Set DISABLE_GEMINI=true
        # in your environment to activate. Not intended as a permanent architecture
        # change — remove once API quota/billing is resolved.
        if os.environ.get('DISABLE_GEMINI', 'false').lower() == 'true':
            logger.warning("⚠️ DISABLE_GEMINI is set — skipping Gemini API, using Ollama/local fallback only.")
        else:
            logger.info("☁️ [Priority 2] Attempting remote Gemini API...")
            max_tokens = int(os.environ.get('GEMINI_MAX_OUTPUT_TOKENS', 2048))
            generation_config = {"max_output_tokens": max_tokens}
            if json_mode:
                generation_config["response_mime_type"] = "application/json"
                
            try:
                response_text = self._call_gemini_raw(final_prompt, generation_config)
                if response_text:
                    if json_mode:
                        return clean_and_repair_json(response_text)
                    return response_text
            except Exception as e:
                logger.warning(f"⚠️ Gemini API failed: {e}. Falling back to open-source offline models...")
            
        # 3. Priority 3: Other open-source alternatives as a last-resort fallback (HuggingFace API -> Local CPU Transformers)
        try:
            logger.info("🔄 [Priority 3] Attempting last-resort open-source fallback models...")
            # Try HF API if token is present
            response_text = call_huggingface_api(final_prompt)
            if response_text:
                logger.info("✅ HuggingFace API inference succeeded!")
                if json_mode:
                    return clean_and_repair_json(response_text)
                return response_text
                
            # Try Local CPU Transformers
            response_text = call_local_transformers(final_prompt)
            if response_text:
                logger.info("✅ Local CPU Transformers inference succeeded!")
                if json_mode:
                    return clean_and_repair_json(response_text)
                return response_text
        except Exception as ex:
            logger.error(f"❌ All LLM options failed: {ex}")
            
        return (
            "⏱️ The clinical agent server is currently experiencing high load. "
            "Please try again in a few seconds."
        )


class LLMResponseAgent:
    def __init__(self):
        self.client = LLMClient()
    
    def generate_response(self, context: List[str], query: str, threshold_met: bool = True, max_similarity: float = 0.0) -> MCPMessage:
        # If similarity is very low (below 15%), use the safety response
        if not threshold_met:
            response_text = """I apologize, but your question doesn't seem to be closely related to the content in your uploaded documents. 

To get better results, try asking questions that are more specific to the document content, such as:
• "What are the main topics discussed?"
• "Can you summarize the key points?"
• "What conclusions or recommendations are mentioned?"

If you believe your question should be answerable from the documents, try rephrasing it to be more specific."""
        # If similarity is low but above threshold (15-40%), try to answer but with a disclaimer
        elif max_similarity < 0.4 and max_similarity > 0.0:
            enhanced_prompt = f"""Based on the available document content, I'll try to answer your question, though the relevance might be limited.

Context from documents:
{chr(10).join(context)}

User Question: {query}

Please provide the best possible answer based on the available context. If the context doesn't directly answer the question, explain what information is available and suggest more specific questions that could be better answered from the document content."""
            
            response_text = self.client.call(enhanced_prompt)
        else:
            # High similarity - use normal response
            response_text = self.client.call(query, context=context)
        
        return MCPMessage(
            sender="LLMResponseAgent",
            receiver="UI",
            type="RESPONSE",
            payload={"answer": response_text, "query": query}
        )
