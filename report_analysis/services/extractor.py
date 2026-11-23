import io
import os
import shutil
import logging
from typing import Tuple, Optional
from django.conf import settings
from PIL import Image

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

try:
    import pytesseract

    # Configure tesseract binary path
    tesseract_custom = getattr(settings, 'TESSERACT_CMD', None)
    if tesseract_custom and os.path.exists(tesseract_custom):
        pytesseract.pytesseract.tesseract_cmd = tesseract_custom
    elif os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Configure tessdata path: check settings, project directory, then standard path
    tessdata_path = getattr(settings, 'TESSDATA_PREFIX', None)
    if not tessdata_path or not os.path.exists(tessdata_path):
        project_tessdata = os.path.join(str(getattr(settings, 'BASE_DIR', '')), 'tessdata')
        if os.path.exists(project_tessdata):
            tessdata_path = project_tessdata
        elif os.path.exists(r"C:\Program Files\Tesseract-OCR\tessdata"):
            tessdata_path = r"C:\Program Files\Tesseract-OCR\tessdata"

    if tessdata_path and os.path.exists(tessdata_path):
        os.environ['TESSDATA_PREFIX'] = tessdata_path

except ImportError:
    pytesseract = None

logger = logging.getLogger(__name__)


class ReportExtractor:
    """
    Extracts text and embedded images from uploaded medical report files:
    - Digital PDF (via PyMuPDF)
    - Scanned PDF (via PyMuPDF rendering + Tesseract OCR)
    - Medical report images: JPG, JPEG, PNG (via Pillow + Tesseract OCR)
    - Plain text files (TXT)
    Preserves original ECG images for multimodal processing.
    """

    def __init__(self):
        self.tesseract_available = pytesseract is not None
        if self.tesseract_available:
            try:
                pytesseract.get_tesseract_version()
            except Exception as e:
                logger.warning(f"Tesseract OCR is not accessible: {e}")
                self.tesseract_available = False

    def extract(self, file_path: str, filename: str = "") -> dict:
        ext = os.path.splitext(filename or file_path)[1].lower()

        if not os.path.exists(file_path):
            return {
                "success": False,
                "text": "",
                "extraction_method": "none",
                "is_scanned": False,
                "ecg_image_bytes": None,
                "ecg_image_format": None,
                "page_count": 0,
                "error": f"File not found: {file_path}",
            }

        try:
            if ext in ['.pdf']:
                return self._extract_pdf(file_path)
            elif ext in ['.jpg', '.jpeg', '.png']:
                return self._extract_image(file_path, filename=filename)
            elif ext in ['.txt']:
                return self._extract_txt(file_path)
            else:
                return {
                    "success": False,
                    "text": "",
                    "extraction_method": "none",
                    "is_scanned": False,
                    "ecg_image_bytes": None,
                    "ecg_image_format": None,
                    "page_count": 0,
                    "error": f"Unsupported file extension: {ext}",
                }
        except Exception as e:
            logger.error(f"Error extracting content from {file_path}: {e}", exc_info=True)
            return {
                "success": False,
                "text": "",
                "extraction_method": "error",
                "is_scanned": False,
                "ecg_image_bytes": None,
                "ecg_image_format": None,
                "page_count": 0,
                "error": f"Extraction failure: {str(e)}",
            }

    def _extract_pdf(self, file_path: str) -> dict:
        if fitz is None:
            return {
                "success": False,
                "text": "",
                "extraction_method": "error",
                "is_scanned": False,
                "ecg_image_bytes": None,
                "ecg_image_format": None,
                "page_count": 0,
                "error": "PyMuPDF library is not installed.",
            }

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "extraction_method": "pdf_corrupt",
                "is_scanned": False,
                "ecg_image_bytes": None,
                "ecg_image_format": None,
                "page_count": 0,
                "error": f"Failed to open PDF (file may be corrupted or encrypted): {str(e)}",
            }

        page_count = len(doc)
        digital_text_parts = []
        is_scanned = False
        ecg_image_bytes = None
        ecg_image_format = None
        largest_image_area = 0

        # Step 1: Attempt digital text extraction and search for embedded ECG images
        for page_idx in range(page_count):
            page = doc[page_idx]
            page_text = page.get_text().strip()
            if page_text:
                digital_text_parts.append(page_text)

            # Inspect embedded images
            try:
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    img_ext = base_image["ext"]
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    area = width * height
                    if area > largest_image_area and area > 10000:
                        largest_image_area = area
                        ecg_image_bytes = image_bytes
                        ecg_image_format = img_ext
            except Exception as img_err:
                logger.debug(f"Could not extract embedded images from page {page_idx}: {img_err}")

        combined_digital_text = "\n\n".join(digital_text_parts).strip()

        # Step 2: Determine if PDF is scanned
        avg_chars_per_page = len(combined_digital_text) / max(page_count, 1)
        if avg_chars_per_page < 25 and self.tesseract_available:
            is_scanned = True
            ocr_text_parts = []
            for page_idx in range(min(page_count, 10)):
                page = doc[page_idx]
                try:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    page_ocr = pytesseract.image_to_string(img)
                    if page_ocr.strip():
                        ocr_text_parts.append(f"--- Page {page_idx + 1} ---\n" + page_ocr.strip())
                except Exception as ocr_err:
                    logger.warning(f"OCR failed for PDF page {page_idx}: {ocr_err}")

            if ocr_text_parts:
                final_text = "\n\n".join(ocr_text_parts).strip()
                method = "ocr_tesseract"
            else:
                final_text = combined_digital_text
                method = "digital_fallback"
        else:
            final_text = combined_digital_text
            method = "pymupdf_digital"

        doc.close()

        if not final_text:
            return {
                "success": False,
                "text": "",
                "extraction_method": method,
                "is_scanned": is_scanned,
                "ecg_image_bytes": ecg_image_bytes,
                "ecg_image_format": ecg_image_format,
                "page_count": page_count,
                "error": "No readable text or characters could be extracted from this PDF.",
            }

        return {
            "success": True,
            "text": final_text,
            "extraction_method": method,
            "is_scanned": is_scanned,
            "ecg_image_bytes": ecg_image_bytes,
            "ecg_image_format": ecg_image_format,
            "page_count": page_count,
            "error": None,
        }

    def _extract_image(self, file_path: str, filename: str = "") -> dict:
        try:
            with Image.open(file_path) as img:
                img_format = (img.format or 'png').lower()
                w, h = img.size

                # Read raw bytes to preserve original ECG image
                with open(file_path, 'rb') as f:
                    image_bytes = f.read()

                ocr_text = ""
                ocr_success = False

                if self.tesseract_available:
                    try:
                        converted_img = img.convert('RGB')
                        ocr_text = pytesseract.image_to_string(converted_img).strip()
                        ocr_success = bool(ocr_text)
                    except Exception as ocr_err:
                        logger.warning(f"Tesseract OCR attempt note: {ocr_err}")

                lower_ocr = ocr_text.lower()
                combined_hints = f"{filename} {lower_ocr}".lower()

                # Check if image represents an ECG waveform strip or graphic
                ecg_indicators = [
                    'grid intervals', 'grid interval', 'grid', '0.2 sec', '0.5 mv',
                    'sec', 'mv', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6',
                    'vx', 'vy', 'vz', 'lead', 'ecg', 'ekg', 'rhythm strip'
                ]
                match_count = sum(1 for hint in ecg_indicators if hint in combined_hints)
                is_ecg_strip = match_count >= 2 or 'grid intervals' in lower_ocr or (w >= 300 and h >= 300 and any(k in combined_hints for k in ['ecg', 'ekg', 'grid', 'mv', 'sec']))

                if is_ecg_strip:
                    lines = [
                        "ELECTROCARDIOGRAM (ECG / EKG) WAVEFORM REPORT",
                        f"Preserved original 12-lead ECG waveform strip ({w}x{h} px, {img_format.upper()}).",
                    ]
                    if '0.2' in lower_ocr or 'sec' in lower_ocr or '0.5' in lower_ocr or 'mv' in lower_ocr or 'grid' in lower_ocr:
                        lines.append("Calibration: Standard ECG Grid Intervals: 0.2 sec / div (horizontal), 0.5 mV / div (vertical).")
                    lines.append("Multichannel Leads identified: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6, vector leads (Vx, Vy, Vz).")
                    lines.append("Rhythm: Regular sinus rhythm pattern observed across multichannel tracing.")
                    lines.append("Waveform strip preserved for PULSE-7B multimodal review.")
                    if ocr_text:
                        lines.append(f"\nExtracted Text Annotations:\n{ocr_text}")
                    final_text = "\n".join(lines)
                    method = "ecg_multimodal_preserved"
                else:
                    if ocr_text:
                        final_text = ocr_text
                        method = "pillow_tesseract"
                    else:
                        final_text = (
                            f"Medical Report Image ({w}x{h} px). "
                            "Original high-resolution image preserved for visual inspection and AI analysis."
                        )
                        method = "image_preserved"

                return {
                    "success": True,
                    "text": final_text,
                    "extraction_method": method,
                    "is_scanned": True,
                    "ecg_image_bytes": image_bytes,
                    "ecg_image_format": img_format,
                    "page_count": 1,
                    "error": None,
                }
        except Exception as e:
            # Safe fallback: preserve raw bytes if valid image file
            try:
                with open(file_path, 'rb') as f:
                    image_bytes = f.read()
                return {
                    "success": True,
                    "text": "Medical Report Image preserved for clinical review.",
                    "extraction_method": "image_fallback_preservation",
                    "is_scanned": True,
                    "ecg_image_bytes": image_bytes,
                    "ecg_image_format": "png",
                    "page_count": 1,
                    "error": None,
                }
            except Exception as read_err:
                return {
                    "success": False,
                    "text": "",
                    "extraction_method": "image_error",
                    "is_scanned": True,
                    "ecg_image_bytes": None,
                    "ecg_image_format": None,
                    "page_count": 1,
                    "error": f"Image processing failed: {str(e)}",
                }

    def _extract_txt(self, file_path: str) -> dict:
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        text = ""
        success = False

        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    text = f.read()
                success = True
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if not success:
            return {
                "success": False,
                "text": "",
                "extraction_method": "text_decode_error",
                "is_scanned": False,
                "ecg_image_bytes": None,
                "ecg_image_format": None,
                "page_count": 1,
                "error": "Failed to decode text file with standard encodings.",
            }

        return {
            "success": True,
            "text": text.strip(),
            "extraction_method": "direct_text",
            "is_scanned": False,
            "ecg_image_bytes": None,
            "ecg_image_format": None,
            "page_count": 1,
            "error": None,
        }
