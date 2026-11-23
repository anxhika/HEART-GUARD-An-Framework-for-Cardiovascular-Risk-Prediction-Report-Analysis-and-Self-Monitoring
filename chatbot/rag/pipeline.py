"""
Main RAG pipeline orchestrator.
This is the single entry point called by the Django view.
"""
from .emergency_detector import is_emergency, get_emergency_response
from .retriever import retrieve
from .prompt_builder import build_prompt
from .llm_client import generate_stream, OllamaConnectionError, OllamaModelNotFoundError


def run_pipeline(
    user_query: str,
    prediction_context: str = "",
    report_context: str = "",
    history: list = None,
) -> dict:
    """
    Orchestrates the full RAG flow.

    Returns:
        {
            "is_emergency": bool,
            "response": str | generator,  # str for emergency, generator for normal
            "sources": list[str],
            "error": str | None,
        }
    """
    if history is None:
        history = []

    # Step 1: Emergency check — bypass LLM entirely
    if is_emergency(user_query):
        return {
            "is_emergency": True,
            "response": get_emergency_response(),
            "sources": [],
            "error": None,
        }

    # Step 2: Retrieve relevant guideline chunks
    retrieval = retrieve(user_query)

    # Step 3: Build the full prompt
    prompt = build_prompt(
        user_query=user_query,
        retrieved_chunks=retrieval["chunks"],
        sources=retrieval["sources"],
        prediction_context=prediction_context,
        report_context=report_context,
        history=history,
    )

    # Step 4: Stream from Ollama
    try:
        token_stream = generate_stream(prompt)
        return {
            "is_emergency": False,
            "response": token_stream,
            "sources": retrieval["sources"],
            "error": None,
        }
    except OllamaConnectionError as e:
        return {
            "is_emergency": False,
            "response": None,
            "sources": [],
            "error": f"Ollama connection failed: {str(e)}",
        }
    except OllamaModelNotFoundError as e:
        return {
            "is_emergency": False,
            "response": None,
            "sources": [],
            "error": str(e),
        }
    except Exception as e:
        return {
            "is_emergency": False,
            "response": None,
            "sources": [],
            "error": f"Unexpected error: {str(e)}",
        }
