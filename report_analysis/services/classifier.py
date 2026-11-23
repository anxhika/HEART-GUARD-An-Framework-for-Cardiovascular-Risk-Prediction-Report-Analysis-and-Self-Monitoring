import re
from typing import Tuple, Dict, Any, List


class ReportClassifier:
    """
    Classifies medical reports into recognized clinical modalities:
    - Echocardiogram (Echo)
    - ECG
    - Blood Test
    - MRI
    - CT Scan
    - Stress Test
    - Angiography
    - Other
    """

    REPORT_TYPES = [
        'Echo',
        'ECG',
        'Blood Test',
        'MRI',
        'CT Scan',
        'Stress Test',
        'Angiography',
        'Other',
    ]

    # Keyword patterns mapped to clinical report types
    KEYWORDS: Dict[str, Dict[str, Any]] = {
        'Echo': {
            'strong': [
                r'\bechocardiogram\b', r'\bechocardiography\b', r'\btransthoracic\b', r'\btte\b',
                r'\btransesophageal\b', r'\btee\b', r'\bejection fraction\b', r'\blvef\b',
                r'\bleft ventricular ejection\b', r'\bmitral regurgitation\b', r'\baortic stenosis\b',
                r'\btricuspid\b', r'\bwall motion abnormality\b', r'\blvedd\b', r'\blvesd\b',
                r'\bpericardial effusion\b', r'\bdoppler echocardiographic\b'
            ],
            'medium': [
                r'\becho\b', r'\bventricle\b', r'\batrium\b', r'\bvalvular\b', r'\bseptum\b',
                r'\bdiastolic dysfunction\b', r'\bsystolic function\b'
            ]
        },
        'ECG': {
            'strong': [
                r'\belectrocardiogram\b', r'\belectrocardiograph\b', r'\belectrocardiography\b',
                r'\b12-lead ecg\b', r'\b12 lead ekg\b', r'\b12-lead ekg\b', r'\bekg\b', r'\becg\b',
                r'\bsinus rhythm\b', r'\bqtc interval\b', r'\bpr interval\b', r'\bqrs duration\b',
                r'\brhythm strip\b', r'\bst segment\b', r'\bt wave inversion\b', r'\bbundle branch block\b',
                r'\blbbb\b', r'\brbbb\b', r'\bventricular tachycardia\b', r'\batrial fibrillation\b',
                r'\bventricular extrasystole\b', r'\bpremature ventricular\b',
                r'\bgrid intervals\b', r'\b0\.2\s*sec\b', r'\b0\.5\s*mv\b',
                r'\bwaveform strip\b', r'\bvector leads?\b'
            ],
            'medium': [
                r'\brhythm\b', r'\baxis\b', r'\bp wave\b', r'\bqtc\b', r'\bheart rate\b', r'\bbpm\b',
                r'\bwaveform\b', r'\bmultichannel leads\b', r'\bgrid\b', r'\bmv\b', r'\bsec\b'
            ]
        },
        'Blood Test': {
            'strong': [
                r'\blipid profile\b', r'\blipid panel\b', r'\bserum cholesterol\b', r'\btriglycerides\b',
                r'\bhdl cholesterol\b', r'\bldl cholesterol\b', r'\btroponin\b', r'\btroponin[- ]?[it]\b',
                r'\bhigh sensitivity troponin\b', r'\bhs-crp\b', r'\bck-mb\b', r'\bcreatine kinase\b',
                r'\bbnp\b', r'\bnt-probnp\b', r'\bhba1c\b', r'\bglycated hemoglobin\b',
                r'\bcomplete blood count\b', r'\bcbc\b', r'\bserum creatinine\b', r'\bblood urea\b',
                r'\belectrolytes panel\b', r'\bfasting blood sugar\b', r'\bfbs\b'
            ],
            'medium': [
                r'\bserum\b', r'\bplasma\b', r'\bmg/dl\b', r'\bmmol/l\b', r'\bng/ml\b',
                r'\breference range\b', r'\bbiological reference interval\b', r'\bpathology report\b'
            ]
        },
        'MRI': {
            'strong': [
                r'\bcardiac mri\b', r'\bcardiac magnetic resonance\b', r'\bcmr\b',
                r'\blate gadolinium enhancement\b', r'\blge\b', r'\bt1 mapping\b', r'\bt2 mapping\b',
                r'\bmyocardial perfusion mri\b', r'\bextracellular volume\b', r'\becv\b',
                r'\bmagnetic resonance imaging\b'
            ],
            'medium': [
                r'\bmri\b', r'\bgadolinium\b', r'\bperfusion defect\b', r'\bmyocardial scar\b'
            ]
        },
        'CT Scan': {
            'strong': [
                r'\bcardiac ct\b', r'\bcoronary ct\b', r'\bcomputed tomography\b',
                r'\bcoronary calcium score\b', r'\bagatston\b', r'\bccta\b',
                r'\bcoronary computed tomography angiogram\b', r'\bcalcium score\b',
                r'\bct angiography\b', r'\bcta chest\b'
            ],
            'medium': [
                r'\bct scan\b', r'\bct angiogram\b', r'\bhounsfield\b', r'\bplaque burden\b'
            ]
        },
        'Stress Test': {
            'strong': [
                r'\btreadmill test\b', r'\btmt\b', r'\bexercise stress test\b', r'\bstress test\b',
                r'\bbruce protocol\b', r'\bmetabolic equivalents\b', r'\bmets\b',
                r'\bexercise tolerance test\b', r'\bdobutamine stress\b', r'\bstress echo\b',
                r'\btarget heart rate achieved\b'
            ],
            'medium': [
                r'\bworkload\b', r'\bpeak heart rate\b', r'\bexercise duration\b',
                r'\bangina on exertion\b', r'\bchronotropic\b'
            ]
        },
        'Angiography': {
            'strong': [
                r'\bcoronary angiography\b', r'\bcoronary angiogram\b', r'\bcath lab\b',
                r'\bcardiac catheterization\b', r'\bcoronary arteries\b', r'\blmca\b',
                r'\bleft main\b', r'\blad\b', r'\bleft anterior descending\b',
                r'\blcx\b', r'\bleft circumflex\b', r'\brca\b', r'\bright coronary artery\b',
                r'\bstenosis\b', r'\btimi flow\b', r'\bpercutaneous coronary intervention\b',
                r'\bpci\b', r'\bstent\b', r'\bheparin\b', r'\bangioplasty\b'
            ],
            'medium': [
                r'\bvascular access\b', r'\bfemoral\b', r'\bradial artery\b', r'\bcontrast media\b',
                r'\bfluoroscopy\b'
            ]
        }
    }

    def classify(self, text: str, filename: str = "") -> Tuple[str, float, bool]:
        """
        Analyzes the text and filename to determine the report type.
        Returns:
            (report_type: str, confidence: float, is_uncertain: bool)
        """
        combined = f"{filename} {text}".lower()
        if not combined.strip():
            return "Other", 0.0, True

        scores: Dict[str, float] = {k: 0.0 for k in self.KEYWORDS.keys()}

        for r_type, patterns in self.KEYWORDS.items():
            for pat in patterns['strong']:
                matches = len(re.findall(pat, combined, flags=re.IGNORECASE))
                if matches > 0:
                    scores[r_type] += matches * 3.0

            for pat in patterns['medium']:
                matches = len(re.findall(pat, combined, flags=re.IGNORECASE))
                if matches > 0:
                    scores[r_type] += matches * 1.0

        best_type = "Other"
        best_score = 0.0

        for r_type, score in scores.items():
            if score > best_score:
                best_score = score
                best_type = r_type

        # Threshold logic
        if best_score == 0:
            return "Other", 0.0, True

        # Normalized confidence indicator between 0.0 and 1.0
        confidence = min(round(best_score / 15.0, 2), 0.99)
        # If score is modest (< 4 points), mark classification as uncertain
        is_uncertain = (best_score < 4.0 or confidence < 0.40)

        return best_type, confidence, is_uncertain
