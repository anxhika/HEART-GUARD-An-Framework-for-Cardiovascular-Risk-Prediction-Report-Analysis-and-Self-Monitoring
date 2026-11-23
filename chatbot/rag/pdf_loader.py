"""
Loads PDFs from the guidelines directory using PyMuPDF,
then chunks the text using LangChain's RecursiveCharacterTextSplitter.
"""
import os
import pymupdf as fitz  # PyMuPDF >= 1.24 uses 'pymupdf' namespace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from django.conf import settings


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text page by page from a PDF.
    Returns list of {text, page_number}.
    """
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            if text:
                pages.append({"text": text, "page": page_num + 1})
        doc.close()
    except Exception as e:
        print(f"[PDFLoader] Error reading {pdf_path}: {e}")
    return pages


def load_and_chunk_all() -> list[dict]:
    """
    Load every PDF in CHATBOT_GUIDELINES_DIR, split into chunks.
    Returns list of {id, text, source, page}.
    """
    guidelines_dir = settings.CHATBOT_GUIDELINES_DIR
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    all_chunks = []
    pdf_files = [f for f in os.listdir(guidelines_dir) if f.lower().endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(guidelines_dir, pdf_file)
        source_name = os.path.splitext(pdf_file)[0]  # filename without extension
        print(f"[PDFLoader] Processing: {pdf_file}")

        pages = extract_text_from_pdf(pdf_path)
        for page_data in pages:
            chunks = splitter.split_text(page_data["text"])
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{source_name}_p{page_data['page']}_c{idx}"
                all_chunks.append({
                    "id": chunk_id,
                    "text": chunk,
                    "source": source_name,
                    "page": page_data["page"],
                })

    print(f"[PDFLoader] Total chunks created: {len(all_chunks)}")
    return all_chunks
