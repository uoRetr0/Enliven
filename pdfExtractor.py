from pathlib import Path
from pypdf import PdfReader

def extract_pdf_to_string(pdf_path: str) -> str:
    """
    extracts all text from a pdf file and returns it as one big string."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    try:
        reader = PdfReader(str(path))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except Exception as e:
        raise Exception(f"Error reading PDF: {e}") from e