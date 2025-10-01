import numpy as np
import noisereduce as nr
import gradio as gr
from transformers import pipeline
from scipy.signal import resample  


DEFAULT_ASR_MODEL = "openai/whisper-tiny"
asr = pipeline("automatic-speech-recognition", model=DEFAULT_ASR_MODEL)

TARGET_SR = 16000 

def denoise_and_transcribe(audio):
    if audio is None:
        return "⚠ No audio provided", None

    sr, data = audio
    data = np.array(data, dtype=np.float32)

    # Stereo → mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Normalize
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
    cleaned = cleaned.astype(np.float32)

   
    if sr != TARGET_SR:
        num_samples = int(len(cleaned) * TARGET_SR / sr)
        cleaned = resample(cleaned, num_samples)
        sr = TARGET_SR

    # Transcribe
    try:
        result = asr({"array": cleaned, "sampling_rate": sr})
        text = result.get("text", "").strip()
    except Exception as e:
        text = f"❌ Error during transcription: {e}"

    return text, (sr, cleaned)


with gr.Blocks() as demo:
    gr.Markdown("# 🎙 Meeting Summarizer\nUpload audio → Clean noise → Get transcript")

    with gr.Row():
        audio_in = gr.Audio(label="Upload your audio (wav/mp3)", type="numpy")
        run_btn = gr.Button("Process")

    with gr.Row():
        transcript_out = gr.Textbox(label="Transcript", lines=8)
        cleaned_audio_out = gr.Audio(label="Cleaned Audio", type="numpy")

    run_btn.click(denoise_and_transcribe, inputs=audio_in, outputs=[transcript_out, cleaned_audio_out])


if __name__ == "__main__":   
    demo.launch()
