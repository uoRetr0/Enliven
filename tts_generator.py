import os
import re

from dotenv import load_dotenv
from elevenlabs import ElevenLabs

load_dotenv()


def _safe_filename(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def _voice_ids(elevenlabs: ElevenLabs) -> list[str]:
    response = elevenlabs.voices.get_all()
    return [v.voice_id for v in response.voices]


def _select_voice_id(key: str, voice_ids: list[str]) -> str:
    if not voice_ids:
        raise ValueError("No ElevenLabs voices available.")
    score = sum(ord(ch) for ch in key.lower())
    return voice_ids[score % len(voice_ids)]


def generate_audio_for_segments(
    segments: list[dict],
    output_dir: str = "output/segments",
    voice_map: dict | None = None,
    narrator_voice_id: str | None = None,
    model_id: str = "eleven_multilingual_v2",
):
    """
    Generates audio files per segment and keeps voices consistent.
    Returns (outputs, voice_map, total_characters).
    """
    elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    voice_ids = _voice_ids(elevenlabs)

    if voice_map is None:
        voice_map = {}

    if narrator_voice_id:
        voice_map.setdefault("narrator", narrator_voice_id)
    else:
        voice_map.setdefault("narrator", voice_ids[0] if voice_ids else "")

    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    total_chars = 0

    for index, segment in enumerate(segments):
        speaker_type = segment.get("speaker_type")
        text = segment.get("text", "")
        total_chars += len(text)

        if speaker_type == "narrator":
            speaker_key = "narrator"
        else:
            speaker_key = segment.get("character_name") or "Unknown"

        voice_id = voice_map.get(speaker_key)
        if not voice_id:
            voice_id = _select_voice_id(speaker_key, voice_ids)
            voice_map[speaker_key] = voice_id

        file_name = f"{index:04d}_{_safe_filename(speaker_key)}.mp3"
        output_path = os.path.join(output_dir, file_name)

        audio = elevenlabs.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model_id,
        )

        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        outputs.append(
            {
                "path": output_path,
                "speaker": speaker_key,
                "voice_id": voice_id,
                "characters": len(text),
            }
        )

    return outputs, voice_map, total_chars


def generate_audiobook(
    segments: list[dict],
    output_path: str = "output/audiobook.wav",
    voice_map: dict | None = None,
    narrator_voice_id: str | None = None,
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "pcm_24000",
):
    """
    Generates a single WAV audiobook file from segments.
    Returns (output_path, voice_map, total_characters).
    """
    elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    voice_ids = _voice_ids(elevenlabs)

    if voice_map is None:
        voice_map = {}

    if narrator_voice_id:
        voice_map.setdefault("narrator", narrator_voice_id)
    else:
        voice_map.setdefault("narrator", voice_ids[0] if voice_ids else "")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_chars = 0
    if output_format.startswith("pcm_"):
        import wave

        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)

            for segment in segments:
                speaker_type = segment.get("speaker_type")
                text = segment.get("text", "")
                total_chars += len(text)

                if speaker_type == "narrator":
                    speaker_key = "narrator"
                else:
                    speaker_key = segment.get("character_name") or "Unknown"

                voice_id = voice_map.get(speaker_key)
                if not voice_id:
                    voice_id = _select_voice_id(speaker_key, voice_ids)
                    voice_map[speaker_key] = voice_id

                audio = elevenlabs.text_to_speech.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id=model_id,
                    output_format=output_format,
                )

                for chunk in audio:
                    wav_file.writeframes(chunk)
    else:
        with open(output_path, "wb") as out_file:
            for segment in segments:
                speaker_type = segment.get("speaker_type")
                text = segment.get("text", "")
                total_chars += len(text)

                if speaker_type == "narrator":
                    speaker_key = "narrator"
                else:
                    speaker_key = segment.get("character_name") or "Unknown"

                voice_id = voice_map.get(speaker_key)
                if not voice_id:
                    voice_id = _select_voice_id(speaker_key, voice_ids)
                    voice_map[speaker_key] = voice_id

                audio = elevenlabs.text_to_speech.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id=model_id,
                    output_format=output_format,
                )

                for chunk in audio:
                    out_file.write(chunk)

    return output_path, voice_map, total_chars


def generate_audiobook_mp3(
    segments: list[dict],
    output_path: str = "output/audiobook.mp3",
    voice_map: dict | None = None,
    narrator_voice_id: str | None = None,
    model_id: str = "eleven_multilingual_v2",
):
    return generate_audiobook(
        segments=segments,
        output_path=output_path,
        voice_map=voice_map,
        narrator_voice_id=narrator_voice_id,
        model_id=model_id,
        output_format="mp3_44100_128",
    )
