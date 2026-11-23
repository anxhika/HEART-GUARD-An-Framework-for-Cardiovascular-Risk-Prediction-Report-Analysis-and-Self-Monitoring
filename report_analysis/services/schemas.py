from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


class ClinicalFindingItem(BaseModel):
    parameter_name: str = Field(description="Name of the clinical parameter or measurement")
    value: str = Field(description="Exact observed value as documented in the report")
    numeric_value: Optional[float] = Field(default=None, description="Parsed numeric value if measurable, or null")
    unit: Optional[str] = Field(default="", description="Measurement unit (e.g., %, mg/dL, bpm, ms, mm), or empty string")
    reference_range: Optional[str] = Field(default="", description="Normal reference interval stated in the report, or empty string")
    status: Literal['normal', 'abnormal', 'critical', 'uncertain', 'unknown'] = Field(
        default='unknown',
        description="Clinical status classification"
    )
    flag: Optional[str] = Field(default="", description="Review indicator e.g. HIGH, LOW, NORMAL, ABNORMAL, or empty")
    evidence: Optional[str] = Field(default="", description="Verbatim quote or text snippet from the report proving this finding")
    clinical_significance: Optional[str] = Field(default="", description="Objective clinical meaning or physiological significance")

    @field_validator('value', mode='before')
    @classmethod
    def stringify_value(cls, v):
        if v is None:
            return ""
        return str(v)

    @field_validator('numeric_value', mode='before')
    @classmethod
    def parse_numeric(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)):
            return float(v)
        try:
            # Clean string to extract leading number
            import re
            m = re.search(r"[-+]?\d*\.?\d+", str(v))
            if m:
                return float(m.group(0))
        except Exception:
            pass
        return None


class BaseReportExtractionSchema(BaseModel):
    report_type: str = Field(description="Standard clinical modality type")
    report_date: Optional[str] = Field(default=None, description="Date of the report, or null if not stated")
    patient_demographics: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Patient details mentioned (age, sex, ID) or empty dict"
    )
    parameters: List[ClinicalFindingItem] = Field(
        default_factory=list,
        description="List of clinical measurements and parameters extracted from the text"
    )
    findings_summary: str = Field(
        default="",
        description="Objective, patient-friendly plain text summary of all verified findings"
    )
    clinical_flags: List[str] = Field(
        default_factory=list,
        description="Specific clinical alert tags such as LOW_LVEF, ELEVATED_TROPONIN, HIGH_LDL"
    )
    has_abnormal_findings: bool = Field(
        default=False,
        description="True if any extracted finding is outside the normal physiological range"
    )
    has_critical_findings: bool = Field(
        default=False,
        description="True if any finding indicates immediate clinical concern or critical alert"
    )
    evidence_notes: Optional[str] = Field(
        default="",
        description="Key diagnostic conclusions or impression lines copied directly from the report"
    )


# Specific report type schemas for specialized validation

class EchoReportSchema(BaseReportExtractionSchema):
    report_type: Literal['Echo'] = 'Echo'
    ejection_fraction: Optional[float] = Field(default=None, description="Left ventricular ejection fraction % (LVEF)")
    lv_dimensions: Optional[Dict[str, Any]] = Field(default_factory=dict, description="LVEDD, LVESD measurements")
    valvular_assessment: Optional[Dict[str, str]] = Field(default_factory=dict, description="Mitral, Aortic, Tricuspid status")
    wall_motion: Optional[str] = Field(default=None, description="Regional or global wall motion summary")
    pericardial_effusion: Optional[str] = Field(default=None, description="Present/absent/mild/moderate/severe")


class ECGReportSchema(BaseReportExtractionSchema):
    report_type: Literal['ECG'] = 'ECG'
    heart_rate_bpm: Optional[float] = Field(default=None, description="Heart rate in beats per minute")
    rhythm: Optional[str] = Field(default=None, description="Sinus rhythm, AFib, atrial flutter, etc.")
    pr_interval_ms: Optional[float] = Field(default=None, description="PR interval in ms")
    qrs_duration_ms: Optional[float] = Field(default=None, description="QRS complex duration in ms")
    qtc_ms: Optional[float] = Field(default=None, description="Corrected QT interval in ms")
    st_t_changes: Optional[str] = Field(default=None, description="ST elevation, ST depression, T-wave inversion")


class BloodTestSchema(BaseReportExtractionSchema):
    report_type: Literal['Blood Test'] = 'Blood Test'
    total_cholesterol: Optional[float] = Field(default=None, description="Total cholesterol mg/dL")
    ldl_cholesterol: Optional[float] = Field(default=None, description="LDL cholesterol mg/dL")
    hdl_cholesterol: Optional[float] = Field(default=None, description="HDL cholesterol mg/dL")
    triglycerides: Optional[float] = Field(default=None, description="Serum triglycerides mg/dL")
    troponin: Optional[float] = Field(default=None, description="Troponin I / T ng/mL or pg/mL")
    bnp_or_ntprobnp: Optional[float] = Field(default=None, description="BNP or NT-proBNP level")
    fasting_blood_sugar: Optional[float] = Field(default=None, description="FBS mg/dL")
    hba1c: Optional[float] = Field(default=None, description="HbA1c %")


class StressTestSchema(BaseReportExtractionSchema):
    report_type: Literal['Stress Test'] = 'Stress Test'
    mets_achieved: Optional[float] = Field(default=None, description="Metabolic equivalents achieved")
    peak_heart_rate: Optional[float] = Field(default=None, description="Peak heart rate achieved bpm")
    target_heart_rate_pct: Optional[float] = Field(default=None, description="Percentage of max predicted HR reached")
    ischemic_changes: Optional[str] = Field(default=None, description="ST changes or symptoms during exercise")
    test_result: Optional[str] = Field(default=None, description="Positive / Negative / Inconclusive for inducible ischemia")


class AngiographySchema(BaseReportExtractionSchema):
    report_type: Literal['Angiography'] = 'Angiography'
    lmca_stenosis: Optional[str] = Field(default=None, description="Left Main Coronary Artery stenosis")
    lad_stenosis: Optional[str] = Field(default=None, description="Left Anterior Descending stenosis")
    lcx_stenosis: Optional[str] = Field(default=None, description="Left Circumflex stenosis")
    rca_stenosis: Optional[str] = Field(default=None, description="Right Coronary Artery stenosis")
    timi_flow: Optional[str] = Field(default=None, description="TIMI flow grade")
    recommendation: Optional[str] = Field(default=None, description="Medical management, PCI/Stenting, or CABG")


SCHEMA_MAP = {
    'Echo': EchoReportSchema,
    'ECG': ECGReportSchema,
    'Blood Test': BloodTestSchema,
    'Stress Test': StressTestSchema,
    'Angiography': AngiographySchema,
}


def get_schema_for_type(report_type: str):
    return SCHEMA_MAP.get(report_type, BaseReportExtractionSchema)
