import os
import logging
from typing import Optional
from django.core.files.base import ContentFile
from django.utils import timezone

from ..models import MedicalReport, ReportFinding
from .extractor import ReportExtractor
from .classifier import ReportClassifier
from .adapters import get_adapter_for_type

logger = logging.getLogger(__name__)


def process_medical_report(report_id: int, manual_type_override: Optional[str] = None) -> MedicalReport:
    """
    Main processing pipeline for a MedicalReport:
    1. Extracts text and embedded ECG images (PyMuPDF, Tesseract OCR, Pillow, TXT).
    2. Classifies report type (or applies manual override).
    3. Runs specialized AI model adapter (HeartDX-LM, PULSE-7B, Qwen 2.5-7B, or fallback).
    4. Validates findings against clinical reference ranges.
    5. Saves structured JSON and individual ReportFinding objects.
    6. Updates report status to completed or failed.
    """
    try:
        report = MedicalReport.objects.get(pk=report_id)
    except MedicalReport.DoesNotExist:
        logger.error(f"MedicalReport with id {report_id} does not exist.")
        raise ValueError(f"MedicalReport with id {report_id} does not exist.")

    report.status = 'processing'
    report.error_message = ''
    report.save()

    try:
        file_path = report.file.path
        filename = report.original_filename or os.path.basename(file_path)

        # 1. Text & Image Extraction
        extractor = ReportExtractor()
        extract_result = extractor.extract(file_path, filename=filename)

        if not extract_result.get("success"):
            report.status = 'failed'
            report.error_message = extract_result.get("error") or "Failed to extract readable content from report."
            report.save()
            return report

        report.extracted_text = extract_result.get("text", "")
        report.file_type = extract_result.get("extraction_method", "")

        # Save preserved ECG image if found and not yet attached
        ecg_bytes = extract_result.get("ecg_image_bytes")
        file_ext = os.path.splitext(filename)[1].lower()
        if ecg_bytes and not report.ecg_image:
            ext = extract_result.get("ecg_image_format") or "png"
            img_name = f"preserved_ecg_{report.id}.{ext}"
            report.ecg_image.save(img_name, ContentFile(ecg_bytes), save=False)
        elif file_ext in ['.jpg', '.jpeg', '.png'] and not report.ecg_image:
            report.ecg_image = report.file

        # 2. Report Type Detection
        classifier = ReportClassifier()
        detected_type, confidence, is_uncertain = classifier.classify(
            text=report.extracted_text,
            filename=filename
        )
        report.detected_type = detected_type

        # Use manual override if user explicitly specified, else detected type
        if manual_type_override and manual_type_override in classifier.REPORT_TYPES:
            final_type = manual_type_override
            report.is_type_manual = True
        elif report.is_type_manual and report.report_type in classifier.REPORT_TYPES:
            final_type = report.report_type
        else:
            final_type = detected_type
            report.is_type_manual = False

        report.report_type = final_type

        # 3. Model Adapter Selection & Clinical Information Extraction
        ecg_path = report.ecg_image.path if report.ecg_image else None
        adapter = get_adapter_for_type(final_type)
        analysis_data = adapter.analyze(
            text=report.extracted_text,
            report_type=final_type,
            ecg_image_path=ecg_path
        )

        # 4. Save Structured Data and Findings
        report.structured_json = analysis_data
        report.findings_summary = analysis_data.get("findings_summary", "")
        report.clinical_flags = analysis_data.get("clinical_flags", [])
        report.has_abnormal_findings = analysis_data.get("has_abnormal_findings", False)
        report.has_critical_findings = analysis_data.get("has_critical_findings", False)
        report.model_used = analysis_data.get("model_used", "")
        report.status = 'completed'
        report.save()

        # 5. Populate relational ReportFinding entries
        report.findings.all().delete()  # Clear any previous findings
        findings_to_create = []
        for p in analysis_data.get("parameters", []):
            findings_to_create.append(ReportFinding(
                report=report,
                parameter_name=p.get("parameter_name", "Unknown Parameter")[:255],
                value=str(p.get("value", ""))[:255],
                numeric_value=p.get("numeric_value"),
                unit=p.get("unit", "")[:50] if p.get("unit") else "",
                reference_range=p.get("reference_range", "")[:150] if p.get("reference_range") else "",
                status=p.get("status", "unknown"),
                flag=p.get("flag", "")[:50] if p.get("flag") else "",
                evidence=p.get("evidence", "") or "",
                clinical_significance=p.get("clinical_significance", "") or ""
            ))
        if findings_to_create:
            ReportFinding.objects.bulk_create(findings_to_create)

        return report

    except Exception as e:
        logger.error(f"Error processing report {report_id}: {e}", exc_info=True)
        report.status = 'failed'
        report.error_message = f"Processing error: {str(e)}"
        report.save()
        return report
