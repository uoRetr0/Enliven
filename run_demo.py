from textParser import parse_text
from pdfExtractor import extract_pdf_to_string
from tts_generator import generate_audiobook_mp3

stringToParse = '''Once upon a time a Wolf was lapping at a spring on a hillside, when, looking up, what should he see but a Lamb just
beginning to drink a little lower down. ‘There’s my supper,’
thought he, ‘if only I can find some excuse to seize it.’ Then
he called out to the Lamb, ‘How dare you muddle the water
from which I am drinking?’
‘Nay, master, nay,’ said Lambikin; ‘if the water be muddy
up there, I cannot be the cause of it, for it runs down from
you to me.’
‘Well, then,’ said the Wolf, ‘why did you call me bad
names this time last year?’
‘That cannot be,’ said the Lamb; ‘I am only six months
old.’
‘I don’t care,’ snarled the Wolf; ‘if it was not you it was
your father;’ and with that he rushed upon the poor little
Lamb and ate her all up. But before she died she gasped out
.’Any excuse will serve a tyrant.’'''
segments = parse_text(stringToParse)
'''output_path, voice_map, total_chars = generate_audiobook_mp3(
    segments,
    output_path="output/audiobook.mp3",
)'''

'''print("Saved:", output_path)
print("Total characters:", total_chars)
print("Voice map:", voice_map)'''
print(segments)
