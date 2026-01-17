import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "google/gemini-3-flash-preview"
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

SYSTEM_INSTRUCTIONS = """
You get one passage from a book.

Return JSON: {"segments":[{"speaker_type":"narrator" or "character","character_name":string or null,"text":string}]}

Rules:
- character = ONLY the words inside quotation marks (spoken/thought words only)
- narrator = ONLY the words outside quotation marks (attribution, action, description)
- CRITICAL: Never duplicate text! Each word appears in exactly ONE segment
- CRITICAL: Narrator text must NOT contain any quoted dialogue - only what's OUTSIDE the quotes
- narrator -> character_name = null (ALWAYS)
- thoughts count as character dialogue when in quotes and attributed
- for thoughts, set character_name to the thinker
- character -> use the FULL NAME if known (e.g., "Harry Potter" not just "Harry")
- if character speaker cannot be identified, use "Unknown"
- Split text at quotation mark boundaries
- Do NOT include quotation marks in the text field
- remove non-story boilerplate (headers/footers, publisher lines, copyright notices)
- keep original order
- return ONLY valid JSON

Example input: "Hello," said John, walking slowly. "How are you?"
Example output: {"segments":[{"speaker_type":"character","character_name":"John","text":"Hello,"},{"speaker_type":"narrator","character_name":null,"text":"said John, walking slowly."},{"speaker_type":"character","character_name":"John","text":"How are you?"}]}

WRONG (duplicates dialogue in narrator): {"segments":[{"speaker_type":"narrator","character_name":null,"text":"\"Hello,\" said John"}]}
RIGHT (splits at quote boundary): {"segments":[{"speaker_type":"character","character_name":"John","text":"Hello,"},{"speaker_type":"narrator","character_name":null,"text":"said John"}]}
"""


def _extract_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Find JSON object with balanced braces
        start = text.find("{")
        if start == -1:
            raise

        depth = 0
        end = start
        for i, char in enumerate(text[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        json_str = text[start:end]
        return json.loads(json_str)


def _split_paragraph(paragraph: str, max_chars: int):
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    chunks = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i : i + max_chars])
            continue
        if len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()

    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, max_chars: int = 3000):
    text = text.strip()
    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for piece in _split_paragraph(paragraph, max_chars):
            if len(current) + len(piece) + 2 > max_chars:
                if current:
                    chunks.append(current)
                current = piece
            else:
                current = f"{current}\n\n{piece}".strip()

    if current:
        chunks.append(current)
    return chunks


def _remove_quoted_overlap(narrator_text: str, adjacent_text: str) -> str:
    """Remove quoted dialogue from narrator text if it appears in adjacent segment."""
    if not adjacent_text:
        return narrator_text

    # Find all quoted content in narrator text
    quotes = re.findall(r'["\u201c\u201d]([^"\u201c\u201d]+)["\u201c\u201d]', narrator_text)

    result = narrator_text
    for quote in quotes:
        # If the quote (or similar) appears in adjacent character dialogue, remove it
        if quote.strip() in adjacent_text or adjacent_text.strip() in quote:
            # Remove the quoted portion including surrounding quotes
            result = re.sub(
                rf'["\u201c\u201d]{re.escape(quote)}["\u201c\u201d]\s*',
                '',
                result
            )

    return result.strip()


def _deduplicate_segments(segments: list[dict]) -> list[dict]:
    """Remove quoted text that appears in both narrator and character segments."""
    if not segments:
        return segments

    cleaned = []
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()

        if seg.get("speaker_type") == "narrator" and text:
            # Check previous segment for overlap
            if cleaned:
                prev = cleaned[-1]
                if prev.get("speaker_type") == "character":
                    text = _remove_quoted_overlap(text, prev.get("text", ""))

            # Check next segment for overlap
            if i + 1 < len(segments):
                next_seg = segments[i + 1]
                if next_seg.get("speaker_type") == "character":
                    text = _remove_quoted_overlap(text, next_seg.get("text", ""))

        if text:
            cleaned.append({**seg, "text": text})

    return cleaned


def parse_passage(
    passage: str, model: str = DEFAULT_MODEL, client_override: OpenAI | None = None
):
    prompt = f'PASSAGE:\n"""{passage}"""'

    active_client = client_override or client
    response = active_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content or ""
    data = _extract_json(content)
    return data["segments"]


def parse_text(
    text: str,
    model: str = DEFAULT_MODEL,
    max_chars: int = 3000,
    client_override: OpenAI | None = None,
):
    chunks = chunk_text(text, max_chars=max_chars)
    if not chunks:
        return []

    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_idx = {
            executor.submit(parse_passage, chunk, model, client_override): idx
            for idx, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    segments = []
    for result in results:
        if result:
            segments.extend(result)

    # Post-process to remove duplicated dialogue
    segments = _deduplicate_segments(segments)
    return segments


def count_characters(segments: list[dict]) -> int:
    return sum(len(segment.get("text", "")) for segment in segments)


def parse_text_with_stats(
    text: str,
    model: str = DEFAULT_MODEL,
    max_chars: int = 3000,
    client_override: OpenAI | None = None,
):
    segments = parse_text(
        text,
        model=model,
        max_chars=max_chars,
        client_override=client_override,
    )
    return {
        "segments": segments,
        "total_characters": count_characters(segments),
    }


def parse_text_to_file(
    text: str,
    output_path: str = "output/segments.json",
    model: str = DEFAULT_MODEL,
    max_chars: int = 3000,
    client_override: OpenAI | None = None,
):
    data = parse_text_with_stats(
        text,
        model=model,
        max_chars=max_chars,
        client_override=client_override,
    )
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_path
