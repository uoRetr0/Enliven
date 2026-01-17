import argparse

from pdfExtractor import extract_pdf_to_string
from textParser import parse_text_to_file
from tts_generator import generate_audiobook_mp3


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an audiobook from a PDF.")
    parser.add_argument("pdf_path", help="Path to the input PDF file.")
    parser.add_argument(
        "--output",
        default="output/audiobook.mp3",
        help="Output audiobook file path.",
    )
    args = parser.parse_args()

    string_to_parse = extract_pdf_to_string(args.pdf_path)
    segments_path = parse_text_to_file(string_to_parse)
    output_path, voice_map, total_chars = generate_audiobook_mp3(
        segments_path,
        output_path=args.output,
    )

    print("Saved:", output_path)
    print("Total characters:", total_chars)
    print("Voice map:", voice_map)


if __name__ == "__main__":
    main()

