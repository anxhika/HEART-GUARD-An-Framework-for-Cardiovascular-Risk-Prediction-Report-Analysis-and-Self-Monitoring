"""
Assembles the final prompt for Llama from:
  1. System instructions
  2. Retrieved medical guideline chunks (with source names)
  3. User's personalised health context (prediction history)
  4. Recent conversation history
  5. The user's current query
"""

SYSTEM_PROMPT = """You are HeartGuard AI, a specialised cardiovascular health assistant integrated into the Heart Disease Prediction System (HDPS).

Your role:
- Answer questions about cardiovascular health, heart disease risk factors, symptoms, prevention, and treatment.
- Explain the user's heart disease prediction results in simple, clear language.
- Summarise findings from the user's medical report analysis when available.
- Always cite the medical guideline sources when your answer is based on retrieved documents.
- If you don't know something or it's outside cardiology, say so clearly.

Your rules:
- NEVER provide a diagnosis. Always remind users to consult a qualified cardiologist.
- NEVER make up medical facts. Only use information from the retrieved guidelines or the user's own data.
- Keep answers concise, compassionate, and in plain English.
- If the user asks about something unrelated to cardiovascular health, politely redirect them.
- Do NOT discuss your own architecture, training, or the fact that you are built on Llama."""


def build_prompt(
    user_query: str,
    retrieved_chunks: list,
    sources: list,
    prediction_context: str,
    report_context: str,
    history: list,          # list of {"role": "user"|"assistant", "content": str}
) -> str:
    parts = [SYSTEM_PROMPT, ""]

    # Medical guidelines context
    if retrieved_chunks:
        parts.append("=== RELEVANT MEDICAL GUIDELINES ===")
        for i, chunk in enumerate(retrieved_chunks, 1):
            parts.append(f"[Chunk {i}]: {chunk.strip()}")
        parts.append(f"Sources: {', '.join(sources)}")
        parts.append("")

    # User health context
    if prediction_context:
        parts.append("=== USER'S HEART DISEASE PREDICTION HISTORY ===")
        parts.append(prediction_context)
        parts.append("")

    # Report analysis context (from report_analysis module)
    if report_context:
        parts.append("=== USER'S MEDICAL REPORT ANALYSIS ===")
        parts.append(report_context)
        parts.append("")

    # Conversation history
    if history:
        parts.append("=== CONVERSATION HISTORY ===")
        for msg in history:
            role_label = "User" if msg["role"] == "user" else "HeartGuard AI"
            parts.append(f"{role_label}: {msg['content']}")
        parts.append("")

    # Current query
    parts.append(f"User: {user_query}")
    parts.append("HeartGuard AI:")

    return "\n".join(parts)
