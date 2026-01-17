"""
Interactive CLI for talking to book characters.
Voice-first with streaming audio and interrupt support.
"""

import os
import sys
import struct
import math
import threading
from character_chat import CharacterChat, get_characters_from_audiobook

try:
    import pyaudio
    import speech_recognition as sr
    AUDIO_AVAILABLE = True
except ImportError as e:
    AUDIO_AVAILABLE = False
    IMPORT_ERROR = str(e)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Audio settings
SAMPLE_RATE = 24000  # ElevenLabs PCM output
INPUT_RATE = 16000   # Input sample rate
CHUNK_SIZE = 480     # 30ms frames at 16kHz


def get_audio_energy(data):
    """Calculate RMS energy of audio chunk."""
    count = len(data) // 2
    if count == 0:
        return 0
    shorts = struct.unpack(f"{count}h", data)
    sum_squares = sum(s * s for s in shorts)
    return math.sqrt(sum_squares / count)


class AudioPlayer:
    """Streams audio with interrupt support."""

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.interrupted = False

    def play_stream(self, audio_generator, check_interrupt=None):
        """Play streamed audio. Returns True if completed, False if interrupted."""
        self.interrupted = False

        stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=1024
        )

        try:
            for chunk in audio_generator:
                if self.interrupted:
                    return False
                if check_interrupt and check_interrupt():
                    self.interrupted = True
                    return False
                stream.write(chunk)
            return True
        finally:
            stream.stop_stream()
            stream.close()

    def stop(self):
        self.interrupted = True

    def cleanup(self):
        self.pa.terminate()


class VoiceInput:
    """Voice input with energy-based detection and interrupt monitoring."""

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.recognizer = sr.Recognizer()
        self.energy_threshold = 500  # Adjusted during calibration

        # Interrupt detection
        self.is_speaking = False
        self._monitoring = False
        self._monitor_thread = None

    def calibrate(self):
        """Calibrate energy threshold based on ambient noise."""
        print("[Calibrating...]", end=" ", flush=True)

        stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=INPUT_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        energies = []
        for _ in range(30):  # ~1 second
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            energies.append(get_audio_energy(data))

        stream.stop_stream()
        stream.close()

        avg_energy = sum(energies) / len(energies)
        self.energy_threshold = avg_energy * 2 + 100  # Above ambient + buffer
        print(f"[Ready - threshold: {int(self.energy_threshold)}]")

    def start_interrupt_monitor(self):
        """Monitor for user speech during playback."""
        self._monitoring = True
        self.is_speaking = False

        def monitor():
            stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=INPUT_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )

            speech_frames = 0

            while self._monitoring:
                try:
                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    energy = get_audio_energy(data)

                    if energy > self.energy_threshold * 1.5:  # Higher threshold for interrupt
                        speech_frames += 1
                        if speech_frames > 5:  # ~150ms of loud audio
                            self.is_speaking = True
                    else:
                        speech_frames = max(0, speech_frames - 1)
                except:
                    break

            stream.stop_stream()
            stream.close()

        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()

    def stop_interrupt_monitor(self):
        self._monitoring = False
        self.is_speaking = False

    def check_interrupt(self):
        return self.is_speaking

    def listen(self) -> str | None:
        """Listen for speech with energy-based endpoint detection."""
        stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=INPUT_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        print("\n[Speak]", end=" ", flush=True)

        frames = []
        triggered = False
        silence_count = 0
        max_silence = 50  # ~1.5s of silence to stop

        try:
            while True:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                energy = get_audio_energy(data)

                if not triggered:
                    if energy > self.energy_threshold:
                        triggered = True
                        frames.append(data)
                else:
                    frames.append(data)

                    if energy > self.energy_threshold:
                        silence_count = 0
                    else:
                        silence_count += 1

                    if silence_count > max_silence:
                        break

                    # Max ~20 seconds
                    if len(frames) > 650:
                        break

        finally:
            stream.stop_stream()
            stream.close()

        if not frames:
            print("[No speech]")
            return None

        print("[...]", end=" ", flush=True)

        # Convert to speech_recognition format
        audio_data = b''.join(frames)
        audio = sr.AudioData(audio_data, INPUT_RATE, 2)

        try:
            text = self.recognizer.recognize_google(audio)
            print(f"[{text}]")
            return text
        except sr.UnknownValueError:
            print("[Unclear]")
            return None
        except sr.RequestError as e:
            print(f"[Error: {e}]")
            return None

    def cleanup(self):
        self.stop_interrupt_monitor()
        self.pa.terminate()


def main():
    print("\n" + "=" * 50)
    print("  ENLIVEN - Talk to Book Characters")
    print("=" * 50)

    if not AUDIO_AVAILABLE:
        print(f"\n[Missing: {IMPORT_ERROR}]")
        print("[Run: pip install pyaudio SpeechRecognition]")
        return

    chat = CharacterChat()
    voice = VoiceInput()
    player = AudioPlayer()
    characters = get_characters_from_audiobook()

    print("\nAvailable characters:")
    for i, char in enumerate(characters, 1):
        role = "(Narrator)" if char.is_narrator else "(Main Character)"
        print(f"  {i}. {char.name} {role}")

    while True:
        choice = input("\nSelect character: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(characters):
                break
        except ValueError:
            pass
        print("Invalid.")

    character = characters[idx]
    chat.set_character(character)

    voice.calibrate()

    print(f"\nTalking to {character.name}")
    if character.is_narrator:
        print("Ask about story context, themes, or minor characters")
    else:
        print("Speaking directly with the character")
    print("Say 'quit' to exit | Speak over to interrupt")
    print("-" * 50)

    try:
        while True:
            user_input = voice.listen()

            if not user_input:
                continue

            lower = user_input.lower()
            if any(w in lower for w in ["quit", "exit", "bye"]):
                print(f"\n{character.name} waves goodbye.")
                print(f"\n[Session {chat.costs}]")
                break

            if "cost" in lower:
                print(f"[{chat.costs}]")
                continue

            try:
                print("[Thinking...]", end=" ", flush=True)
                response, audio_stream = chat.chat_with_voice_stream(user_input)
                print(f"\r{character.name}: {response}")
                print(f"[${chat.costs.total_cost:.4f}]")

                voice.start_interrupt_monitor()
                completed = player.play_stream(audio_stream, voice.check_interrupt)
                voice.stop_interrupt_monitor()

                if not completed:
                    print("[Interrupted]")

            except Exception as e:
                print(f"\n[Error: {e}]")

    finally:
        voice.cleanup()
        player.cleanup()


if __name__ == "__main__":
    main()
