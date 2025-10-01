
import os
import numpy as np
import soundfile as sf
import noisereduce as nr
import gradio as gr
from transformers import pipeline

# Use small model (works better on free CPU Spaces)
DEFAULT_ASR_MODEL = "openai/whisper-tiny"
asr = pipeline("automatic-speech-recognition", model=os.environ.get("ASR_MODEL", DEFAULT_ASR_MODEL))


def denoise_and_transcribe(audio):
    """
    audio: tuple (sample_rate, data) or just numpy array depending on Gradio version
    Returns: transcript + cleaned audio
    """
    if audio is None:
        return "⚠️ No audio provided", None

    # Handle Gradio input format
    if isinstance(audio, tuple):
        sr, data = audio
    else:
        # fallback if only array is provided
        sr = 16000
        data = audio

    # Ensure float32
    data = np.array(data, dtype=np.float32)

    # Convert stereo → mono if needed
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Take first 0.5s for noise sample
    noise_len = int(0.5 * sr)
    noise_len = min(noise_len, len(data) // 2)
    noise_clip = data[:noise_len]

    # Noise reduction
    cleaned = nr.reduce_noise(y=data, sr=sr, y_noise=noise_clip)

    # Save cleaned audio
    cleaned_path = "cleaned_audio.wav"
    sf.write(cleaned_path, cleaned, sr)

    # Run ASR
    try:
        result = asr(cleaned_path)
        text = result.get("text", "").strip()
    except Exception as e:
        text = f"❌ Error during transcription: {e}"

    # Return text + cleaned audio (as tuple for Gradio Audio)
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
