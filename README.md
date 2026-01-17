# Enliven

Talk to book characters with AI-powered voice conversations.

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` with your API keys:
```
ELEVENLABS_API_KEY=your_key
OPENROUTER_API_KEY=your_key
```

## Usage

```bash
python cli.py
```

## Integration

The `get_characters_from_audiobook()` function in `character_chat.py` is a stub.
Replace it with your audiobook team's character data.

```python
from character_chat import CharacterChat, Character

# Create character (from audiobook data)
character = Character(
    name="Character Name",
    description="Physical description",
    personality="How they speak and act",
    backstory="Their background and motivations",
    voice_id="elevenlabs_voice_id"
)

# Chat
chat = CharacterChat()
chat.set_character(character)
response, audio_path = chat.chat_with_voice("Why did you do that?")
```
