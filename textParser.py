import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "google/gemini-2.0-flash-001"
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

SYSTEM_INSTRUCTIONS = """
You get one passage from a book.

Return JSON: {"segments":[{"speaker_type":"narrator" or "character","character_name":string or null,"text":string}]}

Rules:
- character = ONLY text inside quotation marks (the actual spoken words)
- narrator = EVERYTHING outside quotation marks, including:
  - Action descriptions ("he said", "she shouted angrily")
  - Attribution tags ("Ron complained", "Hermione replied")
  - Scene descriptions ("his red hair falling into his eyes")
  - Narrative prose between dialogue
- thoughts count as character dialogue when attributed (e.g., "thought he", "thought the Wolf")
- for thoughts, set character_name to the thinker and keep the quoted thought as character text
- narrator -> character_name = null (ALWAYS)
- character -> use the FULL NAME if known (e.g., "Harry Potter" not just "Harry"), otherwise use exactly what the text provides
- if character speaker cannot be identified, use "Unknown"
- Split text at quotation mark boundaries - do NOT combine narrator text with dialogue
- Do NOT include quotation marks in the text field
- keep original order
- return ONLY valid JSON, no text outside JSON

Example input: "Hello," said John, walking slowly. "How are you?"
Example output: {"segments":[{"speaker_type":"character","character_name":"John","text":"Hello,"},{"speaker_type":"narrator","character_name":null,"text":"said John, walking slowly."},{"speaker_type":"character","character_name":"John","text":"How are you?"}]}
"""

def _extract_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Find JSON object with balanced braces
        start = text.find('{')
        if start == -1:
            raise

        depth = 0
        end = start
        for i, char in enumerate(text[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
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


def parse_passage(passage: str, model: str = DEFAULT_MODEL, client_override: OpenAI | None = None):
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

if __name__ == "__main__":
    sample = '''
    Once upon a time a Wolf was lapping at a spring on a hillside, when, looking up, what should he see but a Lamb just beginning to drink a little lower down. ‘There’s my supper,’ thought he, ‘if only I can find some excuse to seize it.’ Then he called out to the Lamb, ‘How dare you muddle the water from which I am drinking?’ ‘Nay, master, nay,’ said Lambikin; ‘if the water be muddy up there, I cannot be the cause of it, for it runs down from you to me.’ ‘Well, then,’ said the Wolf, ‘why did you call me bad names this time last year?’ ‘That cannot be,’ said the Lamb; ‘I am only six months old.’ ‘I don’t care,’ snarled the Wolf; ‘if it was not you it was your father;’ and with that he rushed upon the poor little Lamb and ate her all up. But before she died she gasped out.’Any excuse will serve a tyrant.’
    '''
    print(parse_text_to_file(sample))
