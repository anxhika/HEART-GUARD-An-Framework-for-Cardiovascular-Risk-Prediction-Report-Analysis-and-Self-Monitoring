"""
Builds a plain-text summary of the user's HeartData prediction history
for injection into the LLM prompt.
"""
from predictor.models import HeartData


CP_LABELS = {0: "typical angina", 1: "atypical angina", 2: "non-anginal pain", 3: "asymptomatic"}
SLOPE_LABELS = {0: "upsloping", 1: "flat", 2: "downsloping"}
THAL_LABELS = {0: "normal", 1: "fixed defect", 2: "reversible defect"}


def build_prediction_context(user) -> str:
    """
    Returns a human-readable string of the user's prediction history.
    Returns empty string if no records exist.
    """
    records = HeartData.objects.filter(owner=user).order_by('-date')[:5]  # last 5 predictions
    if not records.exists():
        return ""

    lines = [f"This user has {records.count()} previous heart disease prediction(s)."]
    for i, r in enumerate(records, 1):
        prob_pct = f"{r.probability * 100:.1f}%" if r.probability is not None else "N/A"
        risk = "HIGH" if r.probability and r.probability > 0.5 else "LOW"
        lines.append(
            f"Record {i} (Date: {r.date}): "
            f"Age={r.age}, Sex={'Male' if r.sex == 1 else 'Female'}, "
            f"Chest Pain Type={CP_LABELS.get(int(r.cp), r.cp)}, "
            f"BP={r.trestbps}mmHg, Cholesterol={r.chol}mg/dL, "
            f"Max HR={r.thalach}, "
            f"Prediction Probability={prob_pct} (Risk: {risk})"
        )

    return "\n".join(lines)
