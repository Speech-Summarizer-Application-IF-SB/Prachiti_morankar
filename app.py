import numpy as np
import noisereduce as nr
import gradio as gr
from transformers import pipeline

# Initialize ASR
DEFAULT_ASR_MODEL = "openai/whisper-tiny"
asr = pipeline("automatic-speech-recognition", model=DEFAULT_ASR_MODEL)

def denoise_and_transcribe(audio):
    """
    audio: tuple (sample_rate, data) from Gradio
    Returns: transcript + cleaned audio
    """
    if audio is None:
        return "⚠️ No audio provided", None

    sr, data = audio

    # Convert to float32
    data = np.array(data, dtype=np.float32)

    # Convert stereo to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Normalize to [-1, 1]
    data = data / (np.max(np.abs(data)) + 1e-9)

    # Noise reduction
    try:
        # If first 0.5s are silent, use as noise profile
        noise_len = min(int(0.5 * sr), len(data) // 2)
        noise_clip = data[:noise_len]
        cleaned = nr.reduce_noise(y=data, y_noise=noise_clip, sr=sr)
    except Exception as e:
        cleaned = data
        print("Noise reduction failed:", e)

    # Normalize cleaned audio to [-1, 1]
    cleaned = cleaned / (np.max(np.abs(cleaned)) + 1e-9)
    cleaned = cleaned.astype(np.float32)  # Ensure dtype is float32

    # Transcribe directly from numpy array
    try:
        result = asr({"array": cleaned, "sampling_rate": sr})
        text = result.get("text", "").strip()
    except Exception as e:
        text = f"❌ Error during transcription: {e}"

    # Return tuple for Gradio audio
    return text, (sr, cleaned)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎙 Meeting Summarizer\nUpload audio → Clean noise → Get transcript")

    with gr.Row():
        audio_in = gr.Audio(label="Upload your audio (wav/mp3)", type="numpy")
        run_btn = gr.Button("Process")

    with gr.Row():
        transcript_out = gr.Textbox(label="Transcript", lines=8)
        cleaned_audio_out = gr.Audio(label="Cleaned Audio")

    run_btn.click(denoise_and_transcribe, inputs=audio_in, outputs=[transcript_out, cleaned_audio_out])

if __name__ == "__main__":
    demo.launch()
