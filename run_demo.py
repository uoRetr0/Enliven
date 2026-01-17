from textParser import parse_text
from pdfExtractor import extract_pdf_to_string
from tts_generator import generate_audiobook_mp3

stringToParse = extract_pdf_to_string("./fables-3.pdf")
segments = parse_text(stringToParse)
output_path, voice_map, total_chars = generate_audiobook_mp3(
    segments,
    output_path="output/audiobook.mp3",
)

print("Saved:", output_path)
print("Total characters:", total_chars)
print("Voice map:", voice_map)
