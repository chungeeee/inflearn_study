from openai import OpenAI
from pathlib import Path
import sounddevice as sd
from scipy.io.wavfile import write
from playsound import playsound

client = OpenAI()


def record_audio():

    fs = 44100
    seconds = 3

    print('녹음을 시작합니다.')
    record = sd.rec(int(seconds*fs), samplerate=fs, channels=2)
    sd.wait()
    print('녹음을 종료합니다.')

    audio_input_path = 'audio_input.wav'
    write(audio_input_path, fs, record)

    return audio_input_path


def conn_whisper(audio_input_path):

    audio_file= open(audio_input_path, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
    )
    print(transcription.text)
    text_input = transcription.text

    return text_input


def conn_chatgpt(text_input):
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text_input}
        ]
    )

    print(completion.choices[0].message.content)
    text_output = completion.choices[0].message.content

    return text_output

def conn_tts(text_output):

    speech_file_path = Path(__file__).parent / "audio_output.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text_output
    )

    response.stream_to_file(speech_file_path)
    audio_output_path = str(speech_file_path)
    
    return audio_output_path

def main():
    audio_input_path = record_audio()
    text_input = conn_whisper(audio_input_path)
    text_output = conn_chatgpt(text_input)
    audio_output_path = conn_tts(text_output)
    playsound(str(audio_output_path))
    return


main()