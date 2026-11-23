from django.contrib import admin
from .models import MedicalReport, ReportFinding


class ReportFindingInline(admin.TabularInline):
    model = ReportFinding
    extra = 0
    fields = ('parameter_name', 'value', 'numeric_value', 'unit', 'reference_range', 'status', 'flag')
    readonly_fields = ('parameter_name', 'value', 'numeric_value', 'unit', 'reference_range', 'status', 'flag')


@admin.register(MedicalReport)
class MedicalReportAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'report_type', 'detected_type', 'status', 'has_abnormal_findings', 'created_at')
    list_filter = ('report_type', 'status', 'has_abnormal_findings', 'has_critical_findings', 'created_at')
    search_fields = ('user__username', 'original_filename', 'findings_summary', 'extracted_text')
    readonly_fields = ('created_at', 'updated_at', 'model_used')
    inlines = [ReportFindingInline]


@admin.register(ReportFinding)
class ReportFindingAdmin(admin.ModelAdmin):
    list_display = ('report', 'parameter_name', 'value', 'unit', 'reference_range', 'status', 'flag')
    list_filter = ('status', 'flag')
    search_fields = ('parameter_name', 'value', 'evidence')
