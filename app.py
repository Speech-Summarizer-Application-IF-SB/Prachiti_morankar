import numpy as np
import noisereduce as nr
import gradio as gr
from transformers import pipeline
from scipy.signal import resample

# Whisper model
asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

TARGET_SR = 16000  # Whisper expects 16kHz

def denoise_and_transcribe(audio):
    if audio is None:
        return "⚠️ No audio provided", None

    try:
        sr, data = audio
        data = np.array(data, dtype=np.float32)

        # Stereo → mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Normalize input
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))

        # Noise reduction
        try:
            noise_len = min(int(0.5 * sr), len(data) // 2)
            noise_clip = data[:noise_len]
            cleaned = nr.reduce_noise(y=data, y_noise=noise_clip, sr=sr)
        except Exception as e:
            print("Noise reduction failed:", e)
            cleaned = data

        # Normalize cleaned
        if np.max(np.abs(cleaned)) > 0:
            cleaned = cleaned / np.max(np.abs(cleaned))
        cleaned = cleaned.astype(np.float32).flatten()

        # Resample → 16kHz for Whisper
        if sr != TARGET_SR and len(cleaned) > 0:
            num_samples = int(len(cleaned) * TARGET_SR / sr)
            cleaned = resample(cleaned, num_samples).astype(np.float32)
            sr = TARGET_SR

        # ✅ Prevent empty audio error
        if len(cleaned) == 0:
            return "⚠️ Empty audio after processing", None

        # Transcribe
        result = asr({"array": cleaned, "sampling_rate": sr})
        text = result.get("text", "").strip()

        # Return transcription + cleaned audio
        return text, (sr, cleaned)

    except Exception as e:
        return f"❌ Runtime error: {e}", None


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎙 Meeting Summarizer\nRecord or upload → Noise removed → Transcript")

    with gr.Row():
        audio_in = gr.Audio(
            label="🎤 Upload or Record", 
            type="numpy", 
            source="microphone"  # ✅ mic recording works
        )
        run_btn = gr.Button("Process")

    with gr.Row():
        transcript_out = gr.Textbox(label="Transcript", lines=8)
        cleaned_audio_out = gr.Audio(label="🔊 Cleaned Audio", type="numpy")

    run_btn.click(denoise_and_transcribe, inputs=audio_in, outputs=[transcript_out, cleaned_audio_out])

if __name__ == "__main__":
    demo.launch()
