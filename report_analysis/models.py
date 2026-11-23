import os
from django.db import models
from django.contrib.auth import get_user_model
from django.urls import reverse

User = get_user_model()


def report_upload_to(instance, filename):
    user_id = instance.user_id if instance.user_id else 'anonymous'
    return os.path.join('reports', f'user_{user_id}', filename)


def ecg_upload_to(instance, filename):
    user_id = instance.user_id if instance.user_id else 'anonymous'
    return os.path.join('reports', f'user_{user_id}', 'ecg_images', filename)


class MedicalReport(models.Model):
    REPORT_TYPE_CHOICES = [
        ('Echo', 'Echocardiogram (Echo)'),
        ('ECG', 'Electrocardiogram (ECG)'),
        ('Blood Test', 'Blood Test (Lipid, Cardiac Enzymes, CBC)'),
        ('MRI', 'Cardiac MRI'),
        ('CT Scan', 'Cardiac CT / Calcium Score'),
        ('Stress Test', 'Cardiac Stress Test (TMT)'),
        ('Angiography', 'Coronary Angiography'),
        ('Other', 'Other Medical Report'),
    ]

    STATUS_CHOICES = [
        ('pending', 'Pending Processing'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='medical_reports')
    file = models.FileField(upload_to=report_upload_to)
    original_filename = models.CharField(max_length=255, blank=True)
    file_type = models.CharField(max_length=50, blank=True)  # pdf, image/jpeg, text/plain, etc.
    file_size = models.PositiveIntegerField(default=0)  # bytes

    report_type = models.CharField(
        max_length=50,
        choices=REPORT_TYPE_CHOICES,
        default='Other'
    )
    detected_type = models.CharField(max_length=50, blank=True)  # automatically detected
    is_type_manual = models.BooleanField(default=False)  # true if user explicitly changed/confirmed

    # Preserved ECG image if applicable (multimodal input preservation)
    ecg_image = models.ImageField(upload_to=ecg_upload_to, null=True, blank=True)

    # Text and Model Extraction
    extracted_text = models.TextField(blank=True, default='')
    structured_json = models.JSONField(default=dict, blank=True)
    findings_summary = models.TextField(blank=True, default='')

    # Clinical Review Flags
    clinical_flags = models.JSONField(default=list, blank=True)  # list of flag objects
    has_abnormal_findings = models.BooleanField(default=False)
    has_critical_findings = models.BooleanField(default=False)

    # Processing metadata
    model_used = models.CharField(max_length=100, blank=True, default='')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    error_message = models.TextField(blank=True, default='')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Medical Report'
        verbose_name_plural = 'Medical Reports'

    def __str__(self):
        return f"{self.report_type} - {self.original_filename or self.id} ({self.user.username})"

    def get_absolute_url(self):
        return reverse('report_analysis:detail', kwargs={'pk': self.pk})

    @property
    def extracted_data(self):
        """Backward-compatible alias for existing chatbot context stub."""
        return self.structured_json

    @property
    def summary(self):
        """Backward-compatible alias for existing chatbot context stub."""
        return self.findings_summary


class ReportFinding(models.Model):
    STATUS_CHOICES = [
        ('normal', 'Normal'),
        ('abnormal', 'Abnormal'),
        ('critical', 'Critical'),
        ('uncertain', 'Uncertain / Borderline'),
        ('unknown', 'Unknown / Not Specified'),
    ]

    report = models.ForeignKey(MedicalReport, on_delete=models.CASCADE, related_name='findings')
    parameter_name = models.CharField(max_length=255)
    value = models.CharField(max_length=255)
    numeric_value = models.FloatField(null=True, blank=True)
    unit = models.CharField(max_length=50, blank=True, default='')
    reference_range = models.CharField(max_length=150, blank=True, default='')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='unknown')
    flag = models.CharField(max_length=50, blank=True, default='')  # HIGH, LOW, NORMAL, ABNORMAL, CRITICAL
    evidence = models.TextField(blank=True, default='')
    clinical_significance = models.TextField(blank=True, default='')

    class Meta:
        ordering = ['id']
        verbose_name = 'Report Finding'
        verbose_name_plural = 'Report Findings'

    def __str__(self):
        return f"{self.parameter_name}: {self.value} ({self.status})"


# Backward compatibility alias for any existing code referencing ReportAnalysis
ReportAnalysis = MedicalReport
