"""
Interactive CLI for talking to book characters.
Voice-first experience - just talk!
"""

import os
import sys
from character_chat import CharacterChat, get_characters_from_audiobook

try:
    import speech_recognition as sr
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def play_audio(filepath: str):
    """Play audio file using system default player."""
    abs_path = os.path.abspath(filepath)
    if sys.platform == "win32":
        os.startfile(abs_path)
    elif sys.platform == "darwin":
        os.system(f'afplay "{abs_path}"')
    else:
        os.system(f'xdg-open "{abs_path}"')


class VoiceInput:
    """Handles microphone input with pre-calibration for speed."""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.energy_threshold = 300  # Lower = more sensitive
        self.recognizer.pause_threshold = 1.5   # Wait longer before cutting off
        self.recognizer.phrase_threshold = 0.3  # Min speech length
        self.recognizer.non_speaking_duration = 1.0  # Silence needed to stop

    def calibrate(self):
        """One-time noise calibration at startup."""
        print("[Calibrating microphone...]")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("[Ready]")

    def listen(self) -> str | None:
        """Listen and return text. Fast, no recalibration."""
        try:
            with sr.Microphone() as source:
                print("\n[Speak]", end=" ", flush=True)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
                print("[...]", end=" ", flush=True)
                text = self.recognizer.recognize_google(audio)
                print(f"[{text}]")
                return text
        except sr.WaitTimeoutError:
            print("[No speech]")
            return None
        except sr.UnknownValueError:
            print("[Unclear]")
            return None
        except sr.RequestError as e:
            print(f"[Error: {e}]")
            return None
        except Exception as e:
            print(f"[Mic error: {e}]")
            return None


def main():
    print("\n" + "=" * 50)
    print("  ENLIVEN - Talk to Book Characters")
    print("=" * 50)

    if not MIC_AVAILABLE:
        print("\n[Warning: speech_recognition not installed]")
        print("[Run: pip install SpeechRecognition pyaudio]")
        return

    # Initialize
    chat = CharacterChat()
    voice = VoiceInput()
    characters = get_characters_from_audiobook()

    print("\nAvailable characters:")
    for i, char in enumerate(characters, 1):
        print(f"  {i}. {char.name}")

    while True:
        choice = input("\nSelect a character (number): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(characters):
                break
        except ValueError:
            pass
        print("Invalid choice.")

    character = characters[idx]
    chat.set_character(character)

    # Calibrate mic once
    voice.calibrate()

    print(f"\nTalking to {character.name}")
    print("Say 'quit' to exit | 'costs' for usage")
    print("-" * 50)

    response_count = 0

    while True:
        user_input = voice.listen()

        if not user_input:
            continue

        # Commands
        lower = user_input.lower()
        if "quit" in lower or "exit" in lower:
            print(f"\n{character.name} waves goodbye.")
            print(f"\n[Session {chat.costs}]")
            break

        if "cost" in lower:
            print(f"[{chat.costs}]")
            continue

        # Chat with voice response
        try:
            response_count += 1
            output_path = os.path.join(OUTPUT_DIR, f"response_{response_count}.mp3")
            response, audio_path = chat.chat_with_voice(user_input, output_path)
            print(f"\n{character.name}: {response}")
            print(f"[${chat.costs.total_cost:.4f}]")
            play_audio(audio_path)

        except Exception as e:
            print(f"\n[Error: {e}]")


if __name__ == "__main__":
    main()
