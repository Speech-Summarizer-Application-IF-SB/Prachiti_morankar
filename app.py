import os
import numpy as np
import soundfile as sf
import noisereduce as nr
import gradio as gr
from transformers import pipeline

# Use Whisper-tiny (fast, works on CPU Spaces)
DEFAULT_ASR_MODEL = "openai/whisper-tiny"

# Load ASR pipeline once
asr = pipeline("automatic-speech-recognition", model=os.environ.get("ASR_MODEL", DEFAULT_ASR_MODEL))

def denoise_and_transcribe(audio_filepath):
    """
    Input: uploaded audio file path
    Output: transcript + cleaned audio
    """
    if not audio_filepath:
        return "⚠️ No audio provided", None

    # Read audio
    data, sr = sf.read(audio_filepath)

    # Convert stereo → mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Take first 0.5s as noise sample
    noise_len = int(0.5 * sr)
    noise_len = min(noise_len, len(data) // 2)
    noise_clip = data[:noise_len]

    # Noise reduction
    cleaned = nr.reduce_noise(y=data, sr=sr, y_noise=noise_clip)

    # Save cleaned file
    cleaned_path = "cleaned_audio.wav"
    sf.write(cleaned_path, cleaned, sr)

    # Run transcription
    try:
        result = asr(cleaned_path)
        text = result.get("text", "").strip()
    except Exception as e:
        text = f"❌ Transcription failed: {e}"

    return text, cleaned_path


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎙 Meeting Summarizer — Milestone 1\nUpload audio → Clean noise → Get transcript")

    with gr.Row():
        audio_in = gr.Audio(type="filepath", label="Upload your audio (wav/mp3)")
        run_btn = gr.Button("Process")

    with gr.Row():
        transcript_out = gr.Textbox(label="Transcript", lines=8)
        cleaned_audio_out = gr.Audio(label="Cleaned Audio")

    run_btn.click(denoise_and_transcribe, inputs=audio_in, outputs=[transcript_out, cleaned_audio_out])


if __name__ == "__main__":
    demo.launch()
