"""
FastAPI server for Enliven - connects frontend to character chat backend
"""

import base64
import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from character_chat import CharacterChat, Character, get_characters_from_audiobook

app = FastAPI(title="Enliven API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory session storage
sessions: dict[str, CharacterChat] = {}
characters_cache: list[Character] = []


class ExtractRequest(BaseModel):
    text: str


class ChatRequest(BaseModel):
    session_id: str
    character_id: str
    message: str


class CharacterResponse(BaseModel):
    id: str
    name: str
    description: str
    personality: str
    backstory: str


class ExtractResponse(BaseModel):
    session_id: str
    characters: list[CharacterResponse]


class ChatResponse(BaseModel):
    text: str
    audio_url: str | None


class CostResponse(BaseModel):
    llm: float
    tts: float
    total: float


class TranscribeRequest(BaseModel):
    audio: str  # base64 encoded raw PCM audio
    sample_rate: int = 16000
    sample_width: int = 2  # 16-bit = 2 bytes


class TranscribeResponse(BaseModel):
    text: str


@app.post("/api/extract-characters", response_model=ExtractResponse)
async def extract_characters(request: ExtractRequest):
    """Extract characters from book text."""
    global characters_cache

    # Get characters (currently returns stub data)
    characters_cache = get_characters_from_audiobook()

    # Create new session
    session_id = str(uuid.uuid4())
    sessions[session_id] = CharacterChat()

    return ExtractResponse(
        session_id=session_id,
        characters=[
            CharacterResponse(
                id=str(i),
                name=char.name,
                description=char.description,
                personality=char.personality,
                backstory=char.backstory,
            )
            for i, char in enumerate(characters_cache)
        ]
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to a character and get response with audio."""
    # Get or create session
    if request.session_id not in sessions:
        sessions[request.session_id] = CharacterChat()

    chat_instance = sessions[request.session_id]

    # Find character
    try:
        char_idx = int(request.character_id)
        if char_idx < 0 or char_idx >= len(characters_cache):
            raise ValueError()
        character = characters_cache[char_idx]
    except (ValueError, IndexError):
        raise HTTPException(status_code=404, detail="Character not found")

    # Set character if changed
    if chat_instance.current_character is None or chat_instance.current_character.name != character.name:
        chat_instance.set_character(character)

    # Get response with voice
    try:
        output_path = f"output/response_{request.session_id}.mp3"
        text_response, audio_path = chat_instance.chat_with_voice(request.message, output_path)

        # Convert audio to base64 data URL
        audio_url = None
        if os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            audio_url = f"data:audio/mpeg;base64,{audio_base64}"
            # Clean up file
            os.remove(audio_path)

        return ChatResponse(text=text_response, audio_url=audio_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/costs", response_model=CostResponse)
async def get_costs(session_id: str = ""):
    """Get current session costs."""
    if session_id and session_id in sessions:
        costs = sessions[session_id].costs
        return CostResponse(
            llm=costs.openrouter_cost,
            tts=costs.elevenlabs_cost,
            total=costs.total_cost,
        )

    # Return aggregate of all sessions
    total_llm = sum(s.costs.openrouter_cost for s in sessions.values())
    total_tts = sum(s.costs.elevenlabs_cost for s in sessions.values())
    return CostResponse(
        llm=total_llm,
        tts=total_tts,
        total=total_llm + total_tts,
    )


@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    """Transcribe raw PCM audio to text using Google Speech Recognition (same as CLI)."""
    import speech_recognition as sr

    try:
        # Decode base64 raw PCM audio
        audio_data = base64.b64decode(request.audio)

        # Create AudioData directly from raw PCM (same as cli.py)
        audio = sr.AudioData(audio_data, request.sample_rate, request.sample_width)

        # Transcribe using Google Web Speech API (free, no API key needed)
        recognizer = sr.Recognizer()
        text = recognizer.recognize_google(audio)

        return TranscribeResponse(text=text)

    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="Could not understand audio")
    except sr.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Speech service error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
