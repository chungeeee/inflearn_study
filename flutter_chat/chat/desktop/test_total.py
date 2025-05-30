from openai import OpenAI
from pathlib import Path

client = OpenAI()


def conn_whisper():

    audio_file= open('audio_input.wav', "rb")
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
    text_input = conn_whisper()
    text_output = conn_chatgpt(text_input)
    audio_output_path = conn_tts(text_output)
    return


main()