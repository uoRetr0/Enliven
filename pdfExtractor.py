import re
from pathlib import Path
from pypdf import PdfReader

# Common boilerplate patterns to remove
BOILERPLATE_PATTERNS = [
    # Website/publisher lines
    r"Free eBooks at Planet eBook\.com",
    r"Planet eBook\.com",
    r"www\.[a-zA-Z0-9-]+\.(com|org|net|edu)",
    r"https?://[^\s]+",
    # Page numbers (standalone numbers or "Page X")
    r"^\s*\d+\s*$",
    r"Page\s+\d+",
    r"^\s*-\s*\d+\s*-\s*$",
    # Copyright notices
    r"Copyright\s*©?\s*\d{4}",
    r"All rights reserved\.?",
    r"Public Domain",
    # Common ebook boilerplate
    r"This eBook is for the use of anyone anywhere",
    r"Project Gutenberg",
    r"Produced by .+",
    r"End of .*Project Gutenberg",
    r"\*\*\*\s*START OF .+\*\*\*",
    r"\*\*\*\s*END OF .+\*\*\*",
    # Table of contents markers
    r"Table of Contents",
    r"CONTENTS",
]


def _clean_boilerplate(text: str) -> str:
    """Remove common boilerplate text from extracted PDF content."""
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # Clean up multiple spaces/newlines left behind
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


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
                # Clean boilerplate from each page
                cleaned = _clean_boilerplate(page_text)
                if cleaned:
                    text_parts.append(cleaned)

        full_text = "\n\n".join(text_parts)

        # Final cleanup pass on the full text
        full_text = _clean_boilerplate(full_text)

        return full_text.strip()
    except Exception as e:
        raise Exception(f"Error reading PDF: {e}") from e