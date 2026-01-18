# Enliven

Talk to book characters with AI-powered voice conversations. Upload any text, extract characters, and have real-time voice conversations with them.

## Features

- **Real-time voice conversations** with ElevenLabs Conversational AI - talk naturally, interrupt anytime
- **Audiobook generation** with word-by-word highlighting and smooth progress tracking
- **Smart character extraction** from any text with auto-generated personalities
- **Intelligent voice assignment** - unique voices per character, gender-aware selection
- **Text formatting** - auto-detects titles, formats dialogue for readability
- Text chat with AI-powered responses
- Cost tracking for API usage

## Setup

### Backend

```bash
pip install -r requirements.txt
```

Create `.env` with your API keys:
```
ELEVENLABS_API_KEY=your_key
OPENROUTER_API_KEY=your_key
```

### Frontend

```bash
cd web
npm install
```

## Usage

### Web UI

Start the backend server:
```bash
python server.py
```

Start the frontend (in a separate terminal):
```bash
cd web
npm run dev
```

Open http://localhost:5173

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/extract-characters` | POST | Extract characters from book text |
| `/api/generate-audiobook` | POST | Generate audiobook with word timestamps |
| `/api/chat` | POST | Send message, get response + audio |
| `/api/transcribe` | POST | Transcribe voice input to text |
| `/api/costs` | GET | Get session API costs |

## Integration

The `get_characters_from_audiobook()` function in `character_chat.py` is a stub. Replace it with your audiobook team's character data.

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

## Tech Stack

- **Backend**: Python, FastAPI, OpenRouter (Gemini Flash), ElevenLabs TTS
- **Frontend**: React, Vite, Tailwind CSS, Framer Motion
- **Voice Input**: Web Audio API with silence detection
