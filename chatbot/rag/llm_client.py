"""
Ollama LLM client using httpx.
Model is fully configurable via Django settings / .env.
Designed so the pipeline never needs to know which model is being used.
"""
import json
import httpx
from django.conf import settings


class OllamaConnectionError(Exception):
    pass

class OllamaModelNotFoundError(Exception):
    pass


def _get_model() -> str:
    return settings.CHATBOT_OLLAMA_MODEL

def _get_url() -> str:
    return settings.CHATBOT_OLLAMA_URL


def check_ollama_available() -> dict:
    """
    Check if Ollama is running and the configured model is available.
    Returns {"available": bool, "model": str, "error": str|None}
    """
    try:
        resp = httpx.get(f"{_get_url()}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        available_models = [m["name"] for m in data.get("models", [])]
        model = _get_model()
        if model not in available_models and not any(model in m for m in available_models):
            return {
                "available": False,
                "model": model,
                "error": f"Model '{model}' not found. Run: ollama pull {model}",
            }
        return {"available": True, "model": model, "error": None}
    except httpx.ConnectError:
        return {
            "available": False,
            "model": _get_model(),
            "error": "Cannot connect to Ollama. Make sure 'ollama serve' is running.",
        }
    except Exception as e:
        return {"available": False, "model": _get_model(), "error": str(e)}


def generate_stream(prompt: str):
    """
    Generator that yields text tokens as they stream from Ollama.
    Raises OllamaConnectionError or OllamaModelNotFoundError on failure.
    """
    model = _get_model()
    url = f"{_get_url()}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": settings.CHATBOT_MAX_TOKENS,
            "temperature": 0.4,
            "top_p": 0.9,
        },
    }
    try:
        with httpx.stream("POST", url, json=payload, timeout=120) as resp:
            if resp.status_code == 404:
                raise OllamaModelNotFoundError(
                    f"Model '{model}' not found on Ollama server. Run: ollama pull {model}"
                )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
    except httpx.ConnectError:
        raise OllamaConnectionError(
            "Cannot connect to Ollama. Make sure 'ollama serve' is running on port 11434."
        )
