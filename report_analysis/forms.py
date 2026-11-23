import os
from django import forms
from django.conf import settings
from .models import MedicalReport


ALLOWED_EXTENSIONS = ['.pdf', '.jpg', '.jpeg', '.png', '.txt']
MAX_SIZE_BYTES = getattr(settings, 'MAX_UPLOAD_SIZE_MB', 15) * 1024 * 1024


def validate_file_extension_and_size(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise forms.ValidationError(
            f"Unsupported file format '{ext}'. Allowed formats: PDF, JPG, PNG, TXT."
        )
    if file.size > MAX_SIZE_BYTES:
        max_mb = MAX_SIZE_BYTES // (1024 * 1024)
        raise forms.ValidationError(
            f"File size ({file.size / (1024 * 1024):.1f} MB) exceeds maximum allowed size of {max_mb} MB."
        )


class ReportUploadForm(forms.ModelForm):
    MANUAL_CHOICES = [('', 'Auto-Detect Report Type')] + MedicalReport.REPORT_TYPE_CHOICES

    manual_report_type = forms.ChoiceField(
        choices=MANUAL_CHOICES,
        required=False,
        widget=forms.Select(attrs={'class': 'browser-default'})
    )

    class Meta:
        model = MedicalReport
        fields = ['file']
        widgets = {
            'file': forms.FileInput(attrs={
                'id': 'report_file_input',
                'accept': '.pdf,.jpg,.jpeg,.png,.txt',
                'class': 'file-input'
            })
        }

    def clean_file(self):
        uploaded_file = self.cleaned_data.get('file')
        if uploaded_file:
            validate_file_extension_and_size(uploaded_file)
        return uploaded_file


class ReportTypeUpdateForm(forms.Form):
    report_type = forms.ChoiceField(
        choices=MedicalReport.REPORT_TYPE_CHOICES,
        widget=forms.Select(attrs={'class': 'browser-default'})
    )
