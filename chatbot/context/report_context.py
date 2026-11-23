"""
Stub for integrating with the report_analysis Django app.
Uses try/except ImportError so the chatbot never crashes
if report_analysis has not been installed yet.
"""


def build_report_context(user) -> str:
    """
    Returns a plain-text summary of the user's latest report analysis.
    Returns empty string if the report_analysis module does not exist
    or if the user has no reports.
    """
    try:
        # This import will succeed once report_analysis is installed.
        # Expected model: ReportAnalysis with fields like:
        #   user (FK), created_at, extracted_data (JSONField), summary (TextField)
        from report_analysis.models import ReportAnalysis  # noqa: F401
        reports = ReportAnalysis.objects.filter(user=user).order_by('-created_at')[:1]
        if not reports.exists():
            return ""
        report = reports.first()
        # Attempt to build a summary from the extracted JSON data
        data = report.extracted_data if hasattr(report, 'extracted_data') else {}
        lines = ["The user's latest medical report analysis extracted the following cardiovascular parameters:"]
        for key, value in data.items():
            lines.append(f"  - {key}: {value}")
        if hasattr(report, 'summary') and report.summary:
            lines.append(f"Clinical Summary: {report.summary}")
        return "\n".join(lines)
    except ImportError:
        # report_analysis app not yet installed — this is expected during development
        return ""
    except Exception:
        # Fail silently — the chatbot should work even if report context fails
        return ""
