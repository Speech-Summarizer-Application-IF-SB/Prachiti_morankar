import os
import tempfile
import numpy as np
import soundfile as sf
import noisereduce as nr
import gradio as gr
from transformers import pipeline

DEFAULT_ASR_MODEL = "facebook/wav2vec2-base-960h"

asr = pipeline("automatic-speech-recognition", model=os.environ.get("ASR_MODEL", DEFAULT_ASR_MODEL))


def denoise_and_transcribe(audio_filepath):
   
    if not audio_filepath:
        return "⚠️ No audio received", None

  
    data, sr = sf.read(audio_filepath)
 
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    noise_len = int(0.5 * sr)
    noise_len = min(noise_len, max(1, len(data)//2))
    noise_clip = data[:noise_len]

    data = data + 1e-9


    cleaned = nr.reduce_noise(y=data, sr=sr, y_noise=noise_clip)

    rms = np.sqrt(np.mean(cleaned**2))
    if rms > 0:
        target_db = -20.0
        rms_db = 20 * np.log10(rms)
        gain = target_db - rms_db
        cleaned = cleaned * (10 ** (gain / 20))


    base = os.path.splitext(os.path.basename(audio_filepath))[0]
    cleaned_path = f"cleaned_{base}.wav"
    sf.write(cleaned_path, cleaned, sr)

    try:
        out = asr(cleaned_path)
        text = out.get("text", "").strip()
    except Exception as e:
        text = f"❌ Transcription failed: {e}"

    return text, cleaned_path


with gr.Blocks(title="Meeting Summarizer (Milestone 1)") as demo:
    gr.Markdown("# 🎙 Meeting Summarizer — Noise reduction + ASR\nUpload or record audio; the app denoises it with `noisereduce` and transcribes with a Hugging Face ASR model.")
    with gr.Row():
        audio_in = gr.Audio(source="upload", type="filepath", label="Upload audio (wav/mp3)")
        btn = gr.Button("Denoise & Transcribe")
    with gr.Row():
        transcript_out = gr.Textbox(label="Transcript", lines=8)
        cleaned_audio_out = gr.Audio(label="Cleaned audio")
    btn.click(denoise_and_transcribe, inputs=audio_in, outputs=[transcript_out, cleaned_audio_out])

if __name__ == "__main__":
    demo.launch()
