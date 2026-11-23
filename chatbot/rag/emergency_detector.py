"""
Detects cardiovascular emergencies in the user's message.

Logic:
  - Only triggers when the user appears to be EXPERIENCING symptoms right now,
    NOT when they are asking an informational question about those symptoms.
  - Informational prefixes (e.g. "what are", "tell me", "explain", "how to")
    suppress the emergency trigger so the RAG pipeline can answer normally.
  - A small set of absolute-trigger phrases (e.g. "i am having a heart attack",
    "call ambulance") always trigger regardless of prefix.
"""

# ── Phrases that ALWAYS trigger, no matter the phrasing ──────────────────────
ABSOLUTE_TRIGGERS = [
    "call ambulance",
    "i am dying",
    "i'm dying",
    "i am having a heart attack",
    "i'm having a heart attack",
    "having a heart attack",
    "i have a cardiac arrest",
    "i'm having a stroke",
    "i am having a stroke",
    "i can't breathe",
    "i cannot breathe",
    "i am not breathing",
    "i'm not breathing",
    "i fainted",
    "i just fainted",
    "i lost consciousness",
    "help me i am",
    "help me i'm",
    "please help i am",
]

# ── Symptom keywords that trigger ONLY when no informational prefix present ───
SYMPTOM_KEYWORDS = [
    "chest pain", "chest tightness", "chest pressure", "crushing pain",
    "can't breathe", "cannot breathe", "difficulty breathing",
    "shortness of breath", "left arm pain", "jaw pain",
    "sudden numbness", "sudden weakness", "sudden vision loss",
    "sudden headache", "sweating profusely", "cold sweat",
    "nausea with chest", "racing heart", "heart racing",
    "rapid heartbeat", "palpitations", "unconscious",
    "lost consciousness", "fainting", "stroke",
]

# ── Informational prefixes that mean the user is ASKING, not experiencing ─────
INFORMATIONAL_PREFIXES = [
    "what are", "what is", "tell me", "explain", "describe",
    "how to", "how do", "how can", "what causes", "symptoms of",
    "signs of", "causes of", "list the", "give me", "show me",
    "can you tell", "do you know", "i want to know", "i want to learn",
    "i want to understand", "information about", "about heart",
    "about cardiac", "about stroke", "about symptoms",
    "prevention of", "treatment of", "how is", "how are",
    "difference between", "define ", "meaning of",
]

EMERGENCY_RESPONSE = """\u26a0\ufe0f EMERGENCY ALERT \u26a0\ufe0f

The symptoms you described may indicate a serious cardiovascular emergency.

\U0001f6a8 PLEASE TAKE IMMEDIATE ACTION:
\u2022 Call emergency services immediately: 112 (India) / 911 (USA)
\u2022 Call a cardiac helpline: 1800-180-1104 (India NPCDCS)
\u2022 Do NOT drive yourself to the hospital.
\u2022 Chew an aspirin (325mg) if not allergic and if available.
\u2022 Sit or lie down and stay calm while waiting for help.
\u2022 Unlock your front door so emergency responders can enter.

This chatbot is NOT a substitute for emergency medical care.
Please seek professional help immediately."""


def is_emergency(text: str) -> bool:
    """
    Returns True only when the user appears to be experiencing an emergency.
    Informational questions (tell me, what are, explain...) return False
    so the RAG pipeline can give a proper educational answer.
    """
    text_lower = text.lower().strip()

    # 1. Absolute triggers — always fire
    if any(phrase in text_lower for phrase in ABSOLUTE_TRIGGERS):
        return True

    # 2. If the message starts with or contains an informational prefix → not an emergency
    if any(prefix in text_lower for prefix in INFORMATIONAL_PREFIXES):
        return False

    # 3. Check symptom keywords — only trigger if no informational context
    return any(keyword in text_lower for keyword in SYMPTOM_KEYWORDS)


def get_emergency_response() -> str:
    """Returns the hardcoded emergency alert message."""
    return EMERGENCY_RESPONSE

