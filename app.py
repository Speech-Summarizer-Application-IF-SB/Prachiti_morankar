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

   
    if data.ndim > 1:
        data = np.mean(data, axis=1)


    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))


    try:
        noise_len = min(int(0.5 * sr), len(data) // 2)
        noise_clip = data[:noise_len]
        cleaned = nr.reduce_noise(y=data, y_noise=noise_clip, sr=sr)
    except Exception as e:
        print("Noise reduction failed:", e)
        cleaned = data

    if np.max(np.abs(cleaned)) > 0:
        cleaned = cleaned / np.max(np.abs(cleaned))
    cleaned = cleaned.astype(np.float32)

  
    if sr != TARGET_SR:
        num_samples = int(len(cleaned) * TARGET_SR / sr)
        cleaned = resample(cleaned, num_samples)
        sr = TARGET_SR


    try:
        result = asr({"array": cleaned, "sampling_rate": sr})
        text = result.get("text", "").strip()
    except Exception as e:
        text = f"❌ Error during transcription: {e}"

    return text, (sr, cleaned)


css = """
body {
    font-family: 'Poppins', sans-serif;
}
.gradio-container {
    background-color: #f0f4f8;
}
#process-btn {
    background-color: #4CAF50 !important;
    color: white !important;
    font-size: 18px;
    border-radius: 12px;
    padding: 12px 25px;
}
.gr-textbox {
    border-radius: 12px;
    border: 1px solid #ccc;
    font-size: 16px;
    padding: 10px;
}
"""


with gr.Blocks(css=css) as demo:
    gr.Markdown(
        "<h1 style='text-align:center'>🎙 Meeting Summarizer</h1>"
        "<p style='text-align:center; font-size:16px;'>Upload or record audio → Clean noise → Get transcript</p>"
    )

    with gr.Row():
        with gr.Column(scale=2):
            audio_in = gr.Audio(label="🎤 Upload or Record Audio", type="numpy")
            run_btn = gr.Button("Process", elem_id="process-btn")
        with gr.Column(scale=3):
            transcript_out = gr.Textbox(label="📝 Transcript", lines=10)
            cleaned_audio_out = gr.Audio(label="🔊 Cleaned Audio", type="numpy")

    run_btn.click(
        denoise_and_transcribe,
        inputs=audio_in,
        outputs=[transcript_out, cleaned_audio_out]
    )

if __name__ == "__main__":
    demo.launch()
