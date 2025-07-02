import speech_recognition as sr
from pydub import AudioSegment

def convert_audio_to_wav(file_path):
    if not file_path.endswith('.wav'):
        audio = AudioSegment.from_file(file_path)
        file_path = file_path.rsplit('.', 1)[0] + '.wav'
        audio.export(file_path, format='wav')
    return file_path

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    file_path = convert_audio_to_wav(file_path)

    with sr.AudioFile(file_path) as source:
        print("Listening...")
        audio = recognizer.record(source)

        try:
            print("Transcribing...")
            text = recognizer.recognize_google(audio)
            print("Transcription:")
            print(text)
            return text
        except sr.UnknownValueError:
            print("Could not understand the audio.")
        except sr.RequestError as e:
            print(f"Google API error: {e}")
        return ""

if __name__ == "__main__":
    text = transcribe_audio("data/sample.wav")
