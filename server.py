"""
FastAPI server for Enliven - connects frontend to character chat backend
"""

import base64
import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from character_chat import CharacterChat, Character, get_characters_from_audiobook, extract_characters_from_text

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


class AudiobookRequest(BaseModel):
    text: str


class WordTiming(BaseModel):
    word: str
    start: float
    end: float


class SegmentAudio(BaseModel):
    speaker: str
    audio_base64: str
    words: list[WordTiming]


class AudiobookResponse(BaseModel):
    segments: list[SegmentAudio]


@app.post("/api/extract-characters", response_model=ExtractResponse)
async def extract_characters(request: ExtractRequest):
    """Extract characters from book text."""
    global characters_cache

    # Extract real characters from text
    characters_cache = extract_characters_from_text(request.text)

    # Fallback if no characters found
    if not characters_cache:
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


# Voice map for audiobook generation (shared across requests)
_audiobook_voice_map: dict[str, str] = {}
_narrator_voice_id: str | None = None


def _get_voice_for_speaker(speaker: str) -> str:
    """Get or assign a voice for a speaker."""
    global _narrator_voice_id
    from elevenlabs import ElevenLabs

    if speaker in _audiobook_voice_map:
        return _audiobook_voice_map[speaker]

    elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    response = elevenlabs.voices.get_all()
    voices = response.voices

    if not voices:
        raise ValueError("No ElevenLabs voices available.")

    # Find a good narrator voice (prefer one labeled for narration/audiobooks)
    if speaker == "narrator":
        if _narrator_voice_id is None:
            # Look for a voice suitable for narration
            narrator_keywords = ["narrator", "audiobook", "story", "neutral"]
            for voice in voices:
                labels = voice.labels or {}
                use_case = labels.get("use_case", "").lower()
                description = labels.get("description", "").lower()
                name_lower = voice.name.lower()

                if any(kw in use_case or kw in description or kw in name_lower for kw in narrator_keywords):
                    _narrator_voice_id = voice.voice_id
                    break

            # Fallback to first voice if no narrator-specific voice found
            if _narrator_voice_id is None:
                _narrator_voice_id = voices[0].voice_id

        _audiobook_voice_map["narrator"] = _narrator_voice_id
        return _narrator_voice_id

    # For characters, exclude the narrator voice and pick from remaining
    voice_ids = [v.voice_id for v in voices if v.voice_id != _narrator_voice_id]

    if not voice_ids:
        # Fallback if all voices were filtered out
        voice_ids = [v.voice_id for v in voices]

    # Deterministic voice selection based on speaker name
    score = sum(ord(ch) for ch in speaker.lower())
    voice_id = voice_ids[score % len(voice_ids)]
    _audiobook_voice_map[speaker] = voice_id

    return voice_id


@app.post("/api/generate-audiobook", response_model=AudiobookResponse)
async def generate_audiobook_endpoint(request: AudiobookRequest):
    """Generate audiobook with word-level timestamps for highlighting."""
    from textParser import parse_text
    from tts_generator import generate_segment_with_timestamps

    try:
        segments = parse_text(request.text)

        result_segments = []
        for seg in segments:
            speaker = seg.get("character_name") or "narrator"
            voice_id = _get_voice_for_speaker(speaker)

            audio_data = generate_segment_with_timestamps(
                text=seg["text"],
                voice_id=voice_id
            )

            result_segments.append(SegmentAudio(
                speaker=speaker,
                audio_base64=audio_data["audio_base64"],
                words=[WordTiming(**w) for w in audio_data["words"]]
            ))

        return AudiobookResponse(segments=result_segments)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
