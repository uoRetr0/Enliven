import re
from pathlib import Path
from pypdf import PdfReader

def extract_pdf_to_string(pdf_path: str) -> str:
    """Extract all text from a PDF file and return it as one clean string."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    try:
        reader = PdfReader(str(path))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                cleaned = re.sub(r"\s+", " ", page_text).strip()
                if cleaned:
                    text_parts.append(cleaned)
        return " ".join(text_parts).strip()
    except Exception as e:
        raise Exception(f"Error reading PDF: {e}") from e