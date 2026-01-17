import pyaudio
import speech_recognition as sr

def test_voice():
    recognizer = sr.Recognizer()
    
    print("Testing microphone...")
    with sr.Microphone() as source:
        print("Say something:")
        audio = recognizer.listen(source, timeout=5)
    
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_voice()