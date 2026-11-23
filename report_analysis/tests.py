import os
import tempfile
from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from .models import MedicalReport, ReportFinding, ReportAnalysis
from .services.classifier import ReportClassifier
from .services.clinical_rules import ClinicalRulesEngine
from .services.extractor import ReportExtractor
from .services.adapters import DeterministicClinicalExtractor, get_adapter_for_type
from .services.pipeline import process_medical_report
from chatbot.context.report_context import build_report_context

User = get_user_model()


class ReportAnalysisModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testcardio', password='password123')

    def test_model_creation_and_alias(self):
        report = MedicalReport.objects.create(
            user=self.user,
            original_filename='echo_test.pdf',
            report_type='Echo',
            detected_type='Echo',
            status='completed',
            findings_summary='Normal left ventricular function.'
        )
        self.assertEqual(report.user.username, 'testcardio')
        self.assertEqual(report.report_type, 'Echo')
        self.assertEqual(str(report), f"Echo - echo_test.pdf ({self.user.username})")

        # Test backward-compatible alias and properties for chatbot
        self.assertTrue(issubclass(ReportAnalysis, MedicalReport) or ReportAnalysis == MedicalReport)
        self.assertEqual(report.summary, 'Normal left ventricular function.')
        self.assertEqual(report.extracted_data, {})

    def test_finding_creation(self):
        report = MedicalReport.objects.create(
            user=self.user,
            original_filename='lab.pdf',
            report_type='Blood Test'
        )
        finding = ReportFinding.objects.create(
            report=report,
            parameter_name='Total Cholesterol',
            value='245 mg/dL',
            numeric_value=245.0,
            unit='mg/dL',
            reference_range='< 200 mg/dL',
            status='abnormal',
            flag='HIGH'
        )
        self.assertEqual(finding.report, report)
        self.assertEqual(finding.status, 'abnormal')
        self.assertEqual(report.findings.count(), 1)


class ClassifierTests(TestCase):
    def setUp(self):
        self.classifier = ReportClassifier()

    def test_echo_classification(self):
        sample = "Transthoracic Echocardiogram Report. Left ventricular ejection fraction (LVEF): 55%. Normal mitral valve."
        report_type, conf, uncertain = self.classifier.classify(sample, filename="echo.pdf")
        self.assertEqual(report_type, "Echo")
        self.assertFalse(uncertain)

    def test_ecg_classification(self):
        sample = "12-Lead Electrocardiogram (ECG/EKG). Normal sinus rhythm, heart rate 72 bpm. PR interval 160 ms, QRS 90 ms."
        report_type, conf, uncertain = self.classifier.classify(sample, filename="ecg.png")
        self.assertEqual(report_type, "ECG")
        self.assertFalse(uncertain)

    def test_blood_classification(self):
        sample = "Pathology Lab Report. Lipid panel: Serum Cholesterol 240 mg/dL, Triglycerides 180 mg/dL, Troponin I: 0.01 ng/mL."
        report_type, conf, uncertain = self.classifier.classify(sample, filename="blood_report.pdf")
        self.assertEqual(report_type, "Blood Test")
        self.assertFalse(uncertain)

    def test_angiography_classification(self):
        sample = "Coronary Angiography Procedure. Left main coronary artery (LMCA) normal. LAD 80% stenosis, TIMI 3 flow."
        report_type, conf, uncertain = self.classifier.classify(sample, filename="cath_lab.pdf")
        self.assertEqual(report_type, "Angiography")

    def test_stress_test_classification(self):
        sample = "Exercise Stress Test (TMT) - Bruce Protocol. Workload achieved 10.2 METs. Target heart rate reached."
        report_type, conf, uncertain = self.classifier.classify(sample, filename="tmt.pdf")
        self.assertEqual(report_type, "Stress Test")


class ClinicalRulesTests(TestCase):
    def setUp(self):
        self.engine = ClinicalRulesEngine()

    def test_normal_lvef(self):
        params = [{'parameter_name': 'LVEF', 'value': '60%', 'numeric_value': 60.0}]
        val_params, flags, has_abnormal, has_critical = self.engine.validate_report(params)
        self.assertFalse(has_abnormal)
        self.assertFalse(has_critical)
        self.assertEqual(val_params[0]['status'], 'normal')

    def test_critical_low_lvef(self):
        params = [{'parameter_name': 'Ejection Fraction', 'value': '25%', 'numeric_value': 25.0}]
        val_params, flags, has_abnormal, has_critical = self.engine.validate_report(params)
        self.assertTrue(has_abnormal)
        self.assertTrue(has_critical)
        self.assertEqual(val_params[0]['status'], 'critical')

    def test_critical_elevated_troponin(self):
        params = [{'parameter_name': 'Cardiac Troponin I', 'value': '0.25 ng/mL', 'numeric_value': 0.25}]
        val_params, flags, has_abnormal, has_critical = self.engine.validate_report(params)
        self.assertTrue(has_abnormal)
        self.assertTrue(has_critical)
        self.assertEqual(val_params[0]['status'], 'critical')

    def test_abnormal_cholesterol(self):
        params = [{'parameter_name': 'Total Cholesterol', 'value': '230 mg/dL', 'numeric_value': 230.0}]
        val_params, flags, has_abnormal, has_critical = self.engine.validate_report(params)
        self.assertTrue(has_abnormal)
        self.assertFalse(has_critical)
        self.assertEqual(val_params[0]['status'], 'abnormal')


class ExtractorAndPipelineTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='docpatient', password='password123')

    def test_txt_extraction_and_pipeline(self):
        content = (
            "ECHOCARDIOGRAM REPORT\n"
            "Date: 2026-05-10\n"
            "Left Ventricular Ejection Fraction (LVEF): 55%\n"
            "Heart Rate: 72 bpm\n"
            "LVEDD: 48 mm\n"
            "Conclusion: Normal LV systolic function, no regional wall motion abnormalities."
        ).encode('utf-8')
        uploaded = SimpleUploadedFile("echo_report.txt", content, content_type="text/plain")

        report = MedicalReport.objects.create(
            user=self.user,
            file=uploaded,
            original_filename="echo_report.txt",
            file_size=len(content)
        )

        processed = process_medical_report(report.id)
        self.assertEqual(processed.status, 'completed')
        self.assertEqual(processed.report_type, 'Echo')
        self.assertTrue(processed.findings.count() >= 2)

    def test_chatbot_context_builder(self):
        # Empty context if no reports
        empty_ctx = build_report_context(self.user)
        self.assertEqual(empty_ctx, "")

        # Create completed report
        report = MedicalReport.objects.create(
            user=self.user,
            file=SimpleUploadedFile("ecg.txt", b"Heart Rate: 80 bpm", content_type="text/plain"),
            original_filename="ecg.txt",
            report_type="ECG",
            status="completed",
            findings_summary="Sinus rhythm at 80 bpm. Normal intervals."
        )
        ReportFinding.objects.create(
            report=report,
            parameter_name="Heart Rate",
            value="80 bpm",
            status="normal",
            flag="NORMAL"
        )

        ctx = build_report_context(self.user)
        self.assertIn("Electrocardiogram", ctx)
        self.assertIn("Sinus rhythm at 80 bpm", ctx)
        self.assertIn("Heart Rate: 80 bpm", ctx)


class ViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user1 = User.objects.create_user(username='user1', password='pass1user')
        self.user2 = User.objects.create_user(username='user2', password='pass2user')

    def test_unauthenticated_redirect(self):
        resp = self.client.get(reverse('report_analysis:history'))
        self.assertEqual(resp.status_code, 302)

    def test_authenticated_history(self):
        self.client.login(username='user1', password='pass1user')
        resp = self.client.get(reverse('report_analysis:history'))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Medical Report Analysis")

    def test_user_data_isolation(self):
        report1 = MedicalReport.objects.create(
            user=self.user1,
            original_filename="user1_report.txt",
            report_type="Echo",
            file=SimpleUploadedFile("user1_report.txt", b"Sample data", content_type="text/plain")
        )
        # Login as user 2 and attempt to view user 1's report
        self.client.login(username='user2', password='pass2user')
        resp = self.client.get(reverse('report_analysis:detail', kwargs={'pk': report1.pk}))
        self.assertEqual(resp.status_code, 404)  # Isolated!
