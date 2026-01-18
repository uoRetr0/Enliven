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
    voice_map: dict[str, str] = field(default_factory=dict)  # Single source of truth for all voices
    voices_cache: list | None = None
    agent_cache: dict[str, str] = field(default_factory=dict)  # character_id -> agent_id cache

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
    audiobook: "AudiobookResponse | None" = None


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


class CreateAgentRequest(BaseModel):
    session_id: str
    character_id: str


class CreateAgentResponse(BaseModel):
    agent_id: str
    signed_url: str


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
    """Extract characters from book text and pre-generate audiobook."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from textParser import parse_text
    from tts_generator import generate_segment_with_timestamps

    # Extract real characters from text
    characters = extract_characters_from_text(request.text)

    # Fallback if no characters found
    if not characters:
        characters = get_characters_from_audiobook()

    # Create new session with session-scoped data
    session_id = str(uuid.uuid4())
    session = SessionData(
        chat=CharacterChat(),
        characters=characters,
    )
    sessions[session_id] = session

    # Pre-generate audiobook so it's ready for instant playback
    audiobook_response = None
    try:
        segments = parse_text(request.text)

        # Pre-assign voices using chat's picker (better quality)
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

        audiobook_response = AudiobookResponse(segments=results)
    except Exception as e:
        print(f"Audiobook pre-generation failed: {e}")
        # Continue without audiobook - user can generate later

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
        ],
        audiobook=audiobook_response,
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

    # Ensure voice consistency: use cached voice from audiobook if available
    char_key = character.name.lower()
    if not character.voice_id and char_key in session.voice_map:
        # Reuse voice from audiobook generation
        character.voice_id = session.voice_map[char_key]

    # Set character for chat (will pick voice if not already set)
    chat_instance.set_character(character)

    # Sync voice to session voice_map
    if character.voice_id:
        session.voice_map[char_key] = character.voice_id

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


@app.post("/api/create-agent", response_model=CreateAgentResponse)
async def create_agent(request: CreateAgentRequest):
    """Get a signed URL for ElevenLabs conversational AI - blazingly fast with caching."""
    import httpx

    # Get session or return error
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please extract characters first.")

    session = sessions[request.session_id]

    # Find character in session-scoped list
    try:
        char_idx = int(request.character_id)
        if char_idx < 0 or char_idx >= len(session.characters):
            raise ValueError()
        character = session.characters[char_idx]
    except (ValueError, IndexError):
        raise HTTPException(status_code=404, detail="Character not found")

    # Ensure character has a voice assigned
    char_key = character.name.lower()
    if not character.voice_id:
        if char_key in session.voice_map:
            character.voice_id = session.voice_map[char_key]
        else:
            session.chat.set_character(character)
            if character.voice_id:
                session.voice_map[char_key] = character.voice_id

    api_key = os.getenv("ELEVENLABS_API_KEY")
    cache_key = f"{request.session_id}:{request.character_id}"

    # Check cache for existing agent
    agent_id = session.agent_cache.get(cache_key)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Create agent if not cached
        if not agent_id:
            system_prompt = f"""You are {character.name} from a story. Stay completely in character.

CHARACTER: {character.name}
Description: {character.description}
Personality: {character.personality}
Backstory: {character.backstory}

RULES:
1. Respond as {character.name} would - use their speech patterns
2. Keep responses brief and conversational (1-2 sentences)
3. You ARE this character. Respond in first person."""

            create_response = await client.post(
                "https://api.elevenlabs.io/v1/convai/agents/create",
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                json={
                    "name": f"enliven-{char_key[:20]}-{request.session_id[:6]}",
                    "conversation_config": {
                        "agent": {
                            "prompt": {"prompt": system_prompt},
                            "first_message": f"Hello! I'm {character.name}. What would you like to talk about?",
                            "language": "en"
                        },
                        "tts": {"voice_id": character.voice_id}
                    }
                }
            )

            if create_response.status_code not in [200, 201]:
                raise HTTPException(status_code=500, detail=f"Agent creation failed: {create_response.text}")

            response_data = create_response.json()
            print(f"[Agent Created] Response: {response_data}")
            agent_id = response_data.get("agent_id")

            if not agent_id:
                raise HTTPException(status_code=500, detail=f"No agent_id in response: {response_data}")

            session.agent_cache[cache_key] = agent_id
            print(f"[Agent Cached] {cache_key} -> {agent_id}")

        # Get signed URL for instant connection
        print(f"[Getting Signed URL] agent_id={agent_id}")
        signed_response = await client.get(
            f"https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id={agent_id}",
            headers={"xi-api-key": api_key}
        )

        if signed_response.status_code != 200:
            # Agent might be invalid, clear cache and retry
            print(f"[Signed URL Failed] Status: {signed_response.status_code}, Response: {signed_response.text}")
            session.agent_cache.pop(cache_key, None)
            raise HTTPException(status_code=500, detail=f"Failed to get signed URL: {signed_response.text}")

        signed_url = signed_response.json().get("signed_url")
        print(f"[Signed URL Success] {signed_url}")
        return CreateAgentResponse(agent_id=agent_id, signed_url=signed_url)


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


# Preferred narrator voice names (in order of preference)
PREFERRED_NARRATOR_NAMES = ["Adam", "Rachel", "Daniel", "Charlotte", "Brian", "Sarah"]


def _score_narrator_voice(voice) -> int:
    """Score how well a voice suits narration (higher = better)."""
    score = 0
    labels = voice.labels or {}
    name = voice.name
    use_case = labels.get("use_case", "").lower()
    description = labels.get("description", "").lower()
    accent = labels.get("accent", "").lower()
    age = labels.get("age", "").lower()

    # Big bonus for preferred premium voices
    if name in PREFERRED_NARRATOR_NAMES:
        position = PREFERRED_NARRATOR_NAMES.index(name)
        score += 200 - (position * 15)

    # Strong preference for narration-focused voices
    if "narration" in use_case or "audiobook" in use_case:
        score += 50
    if "storytelling" in use_case:
        score += 40

    # Voice qualities ideal for narration
    if "clear" in description:
        score += 20
    if "warm" in description:
        score += 15
    if "smooth" in description:
        score += 15
    if "calm" in description or "soothing" in description:
        score += 10

    # Neutral accents work better for narration
    if "american" in accent or "british" in accent:
        score += 10

    # Middle-aged voices tend to work well for narration
    if "middle" in age:
        score += 10

    # Slight penalty for character-focused voices
    if "characters" in use_case and "narration" not in use_case:
        score -= 10

    return score


def _match_speaker_to_character(speaker_key: str, characters: list) -> "Character | None":
    """Find the best matching character for a speaker name.

    Priority:
    1. Exact match
    2. Speaker is substring of character name (e.g., "Jim" in "Jim Smith")
    3. Character name is substring of speaker (e.g., "Wolf" matches "The Wolf")

    When multiple matches, prefer shorter character names (more specific).
    """
    exact_match = None
    substring_matches = []

    for char in characters:
        char_key = char.name.lower()

        # Exact match is highest priority
        if char_key == speaker_key:
            exact_match = char
            break

        # Check both directions for substring matching
        if speaker_key in char_key or char_key in speaker_key:
            substring_matches.append(char)

    if exact_match:
        return exact_match

    if substring_matches:
        # Prefer shorter names (more specific match)
        # e.g., "Jim" over "Jimmy" when speaker is "Jim"
        substring_matches.sort(key=lambda c: len(c.name))
        return substring_matches[0]

    return None


def _get_voice_for_speaker(speaker: str, session: SessionData) -> str:
    """Get or assign a voice for a speaker (session-scoped).

    Uses chat's voice picker as single source of truth for character voices.
    All voice_map keys are normalized to lowercase for consistency.
    """
    speaker_key = speaker.lower().strip()

    # Check cache first (case-insensitive)
    if speaker_key in session.voice_map:
        return session.voice_map[speaker_key]

    # For narrator, find the best available narrator voice
    if speaker_key == "narrator":
        voices = _get_all_voices(session)
        scored = [(v, _score_narrator_voice(v)) for v in voices]
        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][0] if scored else voices[0]
        session.voice_map["narrator"] = best.voice_id
        print(f"[Voice] Narrator → {best.name} (score: {scored[0][1]})")
        return best.voice_id

    # For characters, find best match and use chat's picker
    matched_char = _match_speaker_to_character(speaker_key, session.characters)

    if matched_char:
        char_key = matched_char.name.lower()

        # Ensure voice is assigned (use chat's picker as single source of truth)
        if not matched_char.voice_id:
            session.chat.set_character(matched_char)

        # Store with all relevant keys for consistency
        voice_id = matched_char.voice_id
        session.voice_map[speaker_key] = voice_id
        session.voice_map[char_key] = voice_id
        return voice_id

    # Unknown speaker fallback - use best available narrator voice
    voices = _get_all_voices(session)
    scored = [(v, _score_narrator_voice(v)) for v in voices]
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0][0] if scored else voices[0]
    print(f"[Voice] Unknown speaker '{speaker}' → {best.name} (fallback)")
    session.voice_map[speaker_key] = best.voice_id
    return best.voice_id


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
