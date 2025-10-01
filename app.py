import os
import numpy as np
import soundfile as sf
import noisereduce as nr
import gradio as gr
from transformers import pipeline

# Use a smaller model (fast for Spaces CPU)
DEFAULT_ASR_MODEL = "openai/whisper-tiny"

# Load ASR pipeline
asr = pipeline("automatic-speech-recognition", model=os.environ.get("ASR_MODEL", DEFAULT_ASR_MODEL))


def denoise_and_transcribe(audio_data):
    """
    Input: tuple (numpy array, sample rate) from Gradio
    Output: transcript + cleaned audio
    """
    if audio_data is None:
        return "⚠️ No audio provided", None

    data, sr = audio_data  # Gradio gives (numpy array, sample rate)

    # Convert stereo → mono if needed
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Noise sample from first 0.5s
    noise_len = int(0.5 * sr)
    noise_len = min(noise_len, len(data) // 2)
    noise_clip = data[:noise_len]

    # Noise reduction
    cleaned = nr.reduce_noise(y=data, sr=sr, y_noise=noise_clip)

    # Save cleaned file temporarily
    cleaned_path = "cleaned_audio.wav"
    sf.write(cleaned_path, cleaned, sr)

    # Transcribe
    try:
        result = asr(cleaned_path)
        text = result.get("text", "").strip()
    except Exception as e:
        text = f"❌ Transcription failed: {e}"

    return text, (sr, cleaned)


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎙 Meeting Summarizer — Milestone 1\nUpload audio → Clean noise → Get transcript")

    with gr.Row():
        audio_in = gr.Audio(label="Upload your audio (wav/mp3)")
        run_btn = gr.Button("Process")

    with gr.Row():
        transcript_out = gr.Textbox(label="Transcript", lines=8)
        cleaned_audio_out = gr.Audio(label="Cleaned Audio")

    run_btn.click(denoise_and_transcribe, inputs=audio_in, outputs=[transcript_out, cleaned_audio_out])


if __name__ == "__main__":
    demo.launch()
