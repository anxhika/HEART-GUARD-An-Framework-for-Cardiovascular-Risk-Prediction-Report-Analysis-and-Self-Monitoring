import os
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, Http404
from django.views.decorators.http import require_POST, require_GET

from .models import MedicalReport, ReportFinding
from .forms import ReportUploadForm, ReportTypeUpdateForm
from .services.pipeline import process_medical_report


@login_required
def report_history(request):
    """Lists all medical reports uploaded by the logged-in user."""
    reports = MedicalReport.objects.filter(user=request.user).order_by('-created_at')
    context = {
        'reports': reports,
        'total_count': reports.count(),
        'completed_count': reports.filter(status='completed').count(),
        'abnormal_count': reports.filter(has_abnormal_findings=True).count(),
    }
    return render(request, 'report_analysis/history.html', context)


@login_required
def upload_report(request):
    """
    Handles report file upload via standard form or drag-and-drop AJAX.
    """
    if request.method == 'POST':
        form = ReportUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['file']
            manual_type = form.cleaned_data.get('manual_report_type') or None

            # Create report record
            report = MedicalReport.objects.create(
                user=request.user,
                file=uploaded_file,
                original_filename=uploaded_file.name,
                file_size=uploaded_file.size,
                status='pending',
                report_type=manual_type if manual_type else 'Other',
                is_type_manual=bool(manual_type)
            )

            # Process immediately in this synchronous flow
            process_medical_report(report.id, manual_type_override=manual_type)

            # Support AJAX drag-and-drop responses
            if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.POST.get('ajax') == '1':
                return JsonResponse({
                    'success': True,
                    'report_id': report.id,
                    'redirect_url': report.get_absolute_url()
                })

            messages.success(request, f"Report '{uploaded_file.name}' successfully uploaded and analyzed!")
            return redirect('report_analysis:detail', pk=report.pk)
        else:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.POST.get('ajax') == '1':
                errors = [str(err) for err_list in form.errors.values() for err in err_list]
                return JsonResponse({'success': False, 'error': " ".join(errors)}, status=400)
            messages.error(request, "Failed to upload report. Please check file format and size.")
    else:
        form = ReportUploadForm()

    return render(request, 'report_analysis/upload.html', {'form': form})


@login_required
def report_detail(request, pk):
    """
    Displays full details of an analyzed medical report:
    - Structured parameters
    - Clinical review flags
    - Patient-friendly clinical summary
    - Preserved ECG image viewer
    - Extracted text review
    - Option to re-classify / reprocess
    Strictly isolated to report owner.
    """
    report = get_object_or_404(MedicalReport, pk=pk, user=request.user)
    findings = report.findings.all().order_by('id')
    update_form = ReportTypeUpdateForm(initial={'report_type': report.report_type})

    context = {
        'report': report,
        'findings': findings,
        'update_form': update_form,
        'clinical_flags': report.clinical_flags,
        'structured_data': report.structured_json,
    }
    return render(request, 'report_analysis/detail.html', context)


@login_required
@require_GET
def report_status_api(request, pk):
    """API endpoint to poll the status of a processing report."""
    report = get_object_or_404(MedicalReport, pk=pk, user=request.user)
    return JsonResponse({
        'id': report.id,
        'status': report.status,
        'report_type': report.report_type,
        'has_abnormal': report.has_abnormal_findings,
        'error_message': report.error_message,
        'findings_count': report.findings.count(),
    })


@login_required
@require_POST
def reprocess_report(request, pk):
    """Allows manual override of report type and re-runs analysis."""
    report = get_object_or_404(MedicalReport, pk=pk, user=request.user)
    form = ReportTypeUpdateForm(request.POST)
    if form.is_valid():
        new_type = form.cleaned_data['report_type']
        report.report_type = new_type
        report.is_type_manual = True
        report.save()

        process_medical_report(report.id, manual_type_override=new_type)
        messages.success(request, f"Report re-analyzed as '{new_type}'.")
    else:
        messages.error(request, "Invalid report type selected.")

    return redirect('report_analysis:detail', pk=report.pk)


@login_required
@require_POST
def delete_report(request, pk):
    """Deletes an uploaded report and its associated file storage."""
    report = get_object_or_404(MedicalReport, pk=pk, user=request.user)
    try:
        if report.file and os.path.isfile(report.file.path):
            os.remove(report.file.path)
        if report.ecg_image and os.path.isfile(report.ecg_image.path):
            os.remove(report.ecg_image.path)
    except Exception:
        pass
    report.delete()
    messages.success(request, "Medical report deleted successfully.")
    return redirect('report_analysis:history')
