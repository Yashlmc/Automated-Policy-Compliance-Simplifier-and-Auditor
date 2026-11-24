import os
from pathlib import Path

import fitz


def _read_pdf_bytes(pdf_source):
    """
    Read PDF bytes safely from a Streamlit UploadedFile, file-like object,
    filesystem path (str/Path), or raw bytes. Resets stream cursor first
    so repeated clicks in Streamlit never read empty bytes.
    """
    if isinstance(pdf_source, (bytes, bytearray)):
        return bytes(pdf_source)

    if isinstance(pdf_source, (str, os.PathLike)):
        return Path(pdf_source).read_bytes()

    if hasattr(pdf_source, "seek"):
        try:
            pdf_source.seek(0)
        except Exception:
            pass

    if hasattr(pdf_source, "read"):
        return pdf_source.read()

    raise ValueError("Unsupported PDF input type.")


def extract_pdf_pages(pdf_file):
    """
    Extract page-numbered text from a PDF document.

    Returns:
        List of dicts: [{"page_number": 1, "text": "..."}, ...]
    """
    pdf_bytes = _read_pdf_bytes(pdf_file)
    if not pdf_bytes:
        raise ValueError("The uploaded PDF file is empty.")

    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        if len(pdf) == 0:
            raise ValueError("The PDF contains zero pages.")

        for page_index, page in enumerate(pdf):
            text = page.get_text("text")
            if text and text.strip():
                pages.append(
                    {
                        "page_number": page_index + 1,
                        "text": text.strip(),
                    }
                )

    if not pages:
        raise ValueError(
            "No readable text found in this PDF. "
            "The document may be scanned or image-based and require OCR."
        )

    return pages


def extract_pdf_text(pdf_file):
    """
    Backward-compatible helper returning full document text
    with [PAGE N] markers.
    """
    pages = extract_pdf_pages(pdf_file)
    return "\n".join(
        f"\n[PAGE {page['page_number']}]\n{page['text']}" for page in pages
    )