from openai import OpenAI
from moviepy import VideoFileClip
from dotenv import load_dotenv
import os

# -------------------------
# Load API Key
# -------------------------
load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY")
)

# -------------------------
# Extract Audio
# -------------------------
def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio

    audio_file = "temp_audio.mp3"
    audio.write_audiofile(audio_file)

    video.close()
    audio.close()

    return audio_file


# -------------------------
# Speech to Text
# -------------------------
def transcribe_audio(audio_file):

    with open(audio_file, "rb") as f:

        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f
        )

    return transcript.text


# -------------------------
# Summarize
# -------------------------
def generate_summary(text):

    prompt = f"""
Summarize the following transcript in clear bullet points.

{text}
"""

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt
    )

    return response.output_text


# -------------------------
# Main
# -------------------------
def main(video_path, output_file):

    print("Extracting audio...")
    audio_file = extract_audio(video_path)

    print("Transcribing...")
    transcript = transcribe_audio(audio_file)

    print("Generating summary...")
    summary = generate_summary(transcript)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)

    os.remove(audio_file)

    print("Done!")
    print("Summary saved to:", output_file)


if __name__ == "__main__":

    video_path = "input-version.mp4"
    output_file = "summary.txt"

    main(video_path, output_file)