"""
FastAPI server for Enliven - connects frontend to character chat backend
"""

import base64
import os
import uuid
import tempfile
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from character_chat import CharacterChat, Character, get_characters_from_audiobook, extract_characters_from_text
from pdfExtractor import extract_pdf_to_string


@dataclass
class SessionData:
    """Session-scoped storage for character chat and audiobook state."""
    chat: CharacterChat
    characters: list[Character] = field(default_factory=list)
    voice_map: dict[str, str] = field(default_factory=dict)
    narrator_voice_id: str | None = None
    voices_cache: list | None = None

app = FastAPI(title="Enliven API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory session storage
sessions: dict[str, SessionData] = {}


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
    gender: str


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
    session_id: str | None = None  # Optional: use session for voice mapping


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


class PDFExtractResponse(BaseModel):
    text: str


@app.post("/api/extract-pdf", response_model=PDFExtractResponse)
async def extract_pdf(file: UploadFile = File(...)):
    """Extract text from an uploaded PDF file."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Extract text from PDF
        text = extract_pdf_to_string(tmp_path)

        # Clean up temp file
        os.unlink(tmp_path)

        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF. The PDF may be image-based or empty.")

        return PDFExtractResponse(text=text)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract PDF: {str(e)}")


@app.post("/api/extract-characters", response_model=ExtractResponse)
async def extract_characters(request: ExtractRequest):
    """Extract characters from book text."""
    # Extract real characters from text
    characters = extract_characters_from_text(request.text)

    # Fallback if no characters found
    if not characters:
        characters = get_characters_from_audiobook()

    # Create new session with session-scoped data
    session_id = str(uuid.uuid4())
    sessions[session_id] = SessionData(
        chat=CharacterChat(),
        characters=characters,
    )

    return ExtractResponse(
        session_id=session_id,
        characters=[
            CharacterResponse(
                id=str(i),
                name=char.name,
                description=char.description,
                personality=char.personality,
                backstory=char.backstory,
                gender=char.gender,
            )
            for i, char in enumerate(characters)
        ]
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to a character and get response with audio."""
    # Get session or return error
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please extract characters first.")

    session = sessions[request.session_id]
    chat_instance = session.chat

    # Find character in session-scoped list
    try:
        char_idx = int(request.character_id)
        if char_idx < 0 or char_idx >= len(session.characters):
            raise ValueError()
        character = session.characters[char_idx]
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
        costs = sessions[session_id].chat.costs
        return CostResponse(
            llm=costs.openrouter_cost,
            tts=costs.elevenlabs_cost,
            total=costs.total_cost,
        )

    # Return aggregate of all sessions
    total_llm = sum(s.chat.costs.openrouter_cost for s in sessions.values())
    total_tts = sum(s.chat.costs.elevenlabs_cost for s in sessions.values())
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


def _get_all_voices(session: SessionData | None = None) -> list:
    """Get and cache all ElevenLabs voices (session-scoped if provided)."""
    # Check session cache first
    if session is not None and session.voices_cache is not None:
        return session.voices_cache

    from elevenlabs import ElevenLabs
    elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    response = elevenlabs.voices.get_all()
    voices = response.voices

    # Cache in session if provided
    if session is not None:
        session.voices_cache = voices

    return voices


def _pick_voice_for_character_llm(speaker: str, voices: list, characters: list[Character]) -> str:
    """Use LLM to pick the best voice for a character based on their profile."""
    from openai import OpenAI

    # Find character profile from session characters
    char_profile = None
    for char in characters:
        if char.name.lower() == speaker.lower() or speaker.lower() in char.name.lower():
            char_profile = char
            break

    if not char_profile:
        # No profile found - infer gender from name and pick matching voice
        speaker_lower = speaker.lower()

        # Common female name patterns/endings
        female_patterns = ["ella", "anna", "ina", "ette", "dame", "mrs", "miss", "lady",
                          "della", "bella", "emma", "sophia", "maria", "julia", "sara"]
        # Common male name patterns
        male_patterns = ["jim", "john", "james", "tom", "bob", "bill", "mr", "sir", "lord",
                        "jack", "mike", "david", "peter", "paul", "george", "henry"]

        inferred_gender = "unknown"
        if any(p in speaker_lower for p in female_patterns):
            inferred_gender = "female"
        elif any(p in speaker_lower for p in male_patterns):
            inferred_gender = "male"

        # Filter by inferred gender
        if inferred_gender in ["male", "female"]:
            gender_voices = [v for v in voices if (v.labels or {}).get("gender", "").lower() == inferred_gender]
            if gender_voices:
                score = sum(ord(ch) for ch in speaker_lower)
                return gender_voices[score % len(gender_voices)].voice_id

        # Final fallback: hash-based selection
        voice_ids = [v.voice_id for v in voices]
        score = sum(ord(ch) for ch in speaker_lower)
        return voice_ids[score % len(voice_ids)]

    # Use explicit gender from character profile
    char_gender = char_profile.gender.lower() if char_profile.gender else "unknown"

    # Filter voices by gender first for better matching
    gender_matched_voices = [
        v for v in voices
        if (v.labels or {}).get("gender", "").lower() == char_gender
    ] if char_gender in ["male", "female"] else voices

    # Use gender-matched voices if available, otherwise all voices
    available_voices = gender_matched_voices if gender_matched_voices else voices

    # Build voice descriptions
    voices_desc = "\n".join([
        f"- {v.name} (id: {v.voice_id}): {(v.labels or {}).get('gender', 'unknown')} gender, {(v.labels or {}).get('age', 'unknown')} age, {(v.labels or {}).get('accent', 'unknown')} accent. {(v.labels or {}).get('description', '')}"
        for v in available_voices
    ])

    prompt = f"""Pick the best voice for this character. Return ONLY the voice_id, nothing else.

CHARACTER:
Name: {char_profile.name}
Gender: {char_gender}
Description: {char_profile.description}
Personality: {char_profile.personality}

AVAILABLE VOICES:
{voices_desc}

Return only the voice_id that best matches the character's gender, age, and personality."""

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    response = client.chat.completions.create(
        model="google/gemini-3-flash-preview",
        messages=[{"role": "user", "content": prompt}],
    )

    voice_id = response.choices[0].message.content.strip()

    # Validate voice_id exists
    valid_ids = [v.voice_id for v in voices]
    if voice_id in valid_ids:
        return voice_id

    # Fallback: use first gender-matched voice
    if gender_matched_voices:
        return gender_matched_voices[0].voice_id

    # Final fallback
    return voices[0].voice_id


def _get_voice_for_speaker(speaker: str, session: SessionData) -> str:
    """Get or assign a voice for a speaker (session-scoped)."""
    # Check session voice map cache
    if speaker in session.voice_map:
        return session.voice_map[speaker]

    # Check if character already has a voice_id from chat
    for char in session.characters:
        if char.name.lower() == speaker.lower() or speaker.lower() in char.name.lower():
            if char.voice_id:
                session.voice_map[speaker] = char.voice_id
                return char.voice_id
            break

    voices = _get_all_voices(session)

    if not voices:
        raise ValueError("No ElevenLabs voices available.")

    # Find a good narrator voice (prefer one labeled for narration/audiobooks)
    if speaker == "narrator":
        if session.narrator_voice_id is None:
            # Look for a voice suitable for narration
            narrator_keywords = ["narrator", "audiobook", "story", "neutral"]
            for voice in voices:
                labels = voice.labels or {}
                use_case = labels.get("use_case", "").lower()
                description = labels.get("description", "").lower()
                name_lower = voice.name.lower()

                if any(kw in use_case or kw in description or kw in name_lower for kw in narrator_keywords):
                    session.narrator_voice_id = voice.voice_id
                    break

            # Fallback to first voice if no narrator-specific voice found
            if session.narrator_voice_id is None:
                session.narrator_voice_id = voices[0].voice_id

        session.voice_map["narrator"] = session.narrator_voice_id
        return session.narrator_voice_id

    # For characters, exclude the narrator voice and use LLM to pick best match
    available_voices = [v for v in voices if v.voice_id != session.narrator_voice_id]

    if not available_voices:
        available_voices = voices

    voice_id = _pick_voice_for_character_llm(speaker, available_voices, session.characters)
    session.voice_map[speaker] = voice_id

    # Also update character's voice_id for consistency with chat
    for char in session.characters:
        if char.name.lower() == speaker.lower() or speaker.lower() in char.name.lower():
            char.voice_id = voice_id
            break

    return voice_id


@app.post("/api/generate-audiobook", response_model=AudiobookResponse)
async def generate_audiobook_endpoint(request: AudiobookRequest):
    """Generate audiobook with word-level timestamps for highlighting."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from textParser import parse_text
    from tts_generator import generate_segment_with_timestamps

    try:
        # Get or create session for voice mapping
        if request.session_id and request.session_id in sessions:
            session = sessions[request.session_id]
        else:
            # Create temporary session for this request
            temp_session_id = str(uuid.uuid4())
            session = SessionData(chat=CharacterChat())
            sessions[temp_session_id] = session

        segments = parse_text(request.text)

        # Pre-assign all voices first (sequential to avoid race conditions)
        segment_voices = []
        for seg in segments:
            speaker = seg.get("character_name") or "narrator"
            voice_id = _get_voice_for_speaker(speaker, session)
            segment_voices.append((seg, speaker, voice_id))

        # Generate TTS in parallel
        def generate_audio(args):
            idx, seg, speaker, voice_id = args
            audio_data = generate_segment_with_timestamps(
                text=seg["text"],
                voice_id=voice_id
            )
            return idx, speaker, audio_data

        results = [None] * len(segment_voices)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(generate_audio, (i, seg, speaker, voice_id)): i
                for i, (seg, speaker, voice_id) in enumerate(segment_voices)
            }
            for future in as_completed(futures):
                idx, speaker, audio_data = future.result()
                results[idx] = SegmentAudio(
                    speaker=speaker,
                    audio_base64=audio_data["audio_base64"],
                    words=[WordTiming(**w) for w in audio_data["words"]]
                )

        return AudiobookResponse(segments=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
