"""
Integration bridge between HeartGuard RAG Chatbot and the report_analysis Django app.
Provides user-isolated medical report context for the cardiovascular chatbot.
"""
import logging

logger = logging.getLogger(__name__)


def build_report_context(user) -> str:
    """
    Returns a plain-text summary of the user's latest medical reports and clinical findings.
    Strictly isolated to reports owned by `user`.
    Returns empty string if the user has no reports or if report_analysis fails.
    """
    if not user or not user.is_authenticated:
        return ""

    try:
        from report_analysis.models import MedicalReport

        # Retrieve up to 3 most recent reports for the user
        reports = MedicalReport.objects.filter(
            user=user,
            status='completed'
        ).order_by('-created_at')[:3]

        if not reports.exists():
            return ""

        context_lines = [
            f"The patient has {reports.count()} recent medical report(s) on file:"
        ]

        for idx, report in enumerate(reports, start=1):
            date_str = report.created_at.strftime('%Y-%m-%d')
            context_lines.append(
                f"\n[Report {idx}]: {report.get_report_type_display()} ({report.original_filename or 'File'}), Date: {date_str}"
            )
            if report.findings_summary:
                context_lines.append(f"  Summary: {report.findings_summary}")

            # Include key extracted parameters
            findings = report.findings.all()[:10]
            if findings.exists():
                param_strs = []
                for f in findings:
                    status_flag = f" [{f.flag}]" if f.flag and f.flag != "NORMAL" else ""
                    param_strs.append(f"{f.parameter_name}: {f.value}{status_flag}")
                context_lines.append("  Key Parameters: " + "; ".join(param_strs))

            # Include clinical review flags if any
            if report.clinical_flags:
                flag_messages = []
                for flag in report.clinical_flags:
                    if isinstance(flag, dict):
                        msg = flag.get('message') or f"{flag.get('parameter')}: {flag.get('value')}"
                        flag_messages.append(msg)
                    elif isinstance(flag, str):
                        flag_messages.append(flag)
                if flag_messages:
                    context_lines.append("  Clinical Review Flags: " + " | ".join(flag_messages[:4]))

        return "\n".join(context_lines)

    except ImportError:
        logger.debug("report_analysis module not found when building report context.")
        return ""
    except Exception as e:
        logger.warning(f"Error building report context for user {user}: {e}")
        return ""
