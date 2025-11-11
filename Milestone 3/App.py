import os
import streamlit as st
import sounddevice as sd
import wave
import tempfile
import speech_recognition as sr
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="🎙️ AI Live Meeting Summarizer", layout="wide")

# -------------------- CUSTOM STYLES --------------------
st.markdown("""
<style>
body, .stApp {
    background-color: #f7faff;
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    color: #1E3A8A;
}
hr {
    border: 0;
    border-top: 1px solid #E2E8F0;
    margin: 1.5rem 0;
}
button[kind="primary"] {
    background: linear-gradient(90deg, #2563EB, #60A5FA) !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease-in-out;
}
button[kind="primary"]:hover {
    transform: scale(1.03);
}
.stRadio > div {
    background-color: #EFF6FF;
    border-radius: 10px;
    padding: 10px;
}
textarea {
    border-radius: 10px !important;
    border: 1px solid #CBD5E1 !important;
    background-color: #F9FAFB !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div style='background: linear-gradient(90deg, #3B82F6, #60A5FA);
padding: 1.2rem 2rem; border-radius: 0 0 15px 15px;
box-shadow: 0 3px 10px rgba(0,0,0,0.1); text-align: center;'>
  <h1 style='color: white; font-size: 34px; font-weight: 800; margin-bottom: 0.3rem;'>
    🎙️  AI Live Meeting Summarizer
  </h1>
  <p style='color: #E0F2FE; font-size: 16px;'>
    Record or upload WAV audio → Get instant transcription & summary
  </p>
</div>
""", unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

# -------------------- SIMPLE SUMMARIZER --------------------
def simple_summarizer(text, num_sentences=3):
    """TF-IDF extractive summarizer."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) <= num_sentences:
        return text
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()
    ranked = np.argsort(scores)[::-1][:num_sentences]
    summary = " ".join([sentences[i] for i in sorted(ranked)])
    return summary

# -------------------- SPEECH RECOGNITION --------------------
recognizer = sr.Recognizer()

def transcribe_audio(path):
    """Transcribe audio file using Google Web Speech API."""
    with sr.AudioFile(path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text

# -------------------- AUDIO RECORDING --------------------
def record_audio(duration=5, fs=44100):
    """Record mic audio and save as .wav file."""
    st.info(f"🎤 Recording for {duration} seconds... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmpfile.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())
    st.success("✅ Recording finished! Now click '🚀 Process Audio'.")
    return tmpfile.name

# -------------------- MAIN LAYOUT --------------------
col1, col2 = st.columns([1, 2], gap="large")

# 🎧 Input Section
with col1:
    st.subheader("🎧 Input Options")
    mode = st.radio("Choose Input Type:", ["🎙️ Record from Microphone", "📁 Upload WAV File"])

    if mode == "🎙️ Record from Microphone":
        duration = st.slider("Recording Duration (seconds)", 3, 20, 5)
        if st.button("🎤 Start Recording"):
            st.session_state.audio_path = record_audio(duration)
            st.audio(st.session_state.audio_path)

    elif mode == "📁 Upload WAV File":
        uploaded = st.file_uploader("Upload a .wav file", type=["wav"])
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(uploaded.read())
            st.session_state.audio_path = tmp.name
            st.audio(tmp.name)
            st.success("✅ File uploaded successfully!")

# 🧠 Output Section
with col2:
    st.subheader("🧠 Output")
    st.markdown("<p style='color:#475569;'>Click below to transcribe and summarize your audio.</p>", unsafe_allow_html=True)
    
    if st.session_state.audio_path and st.button("🚀 Transcribe & Summarize"):
        try:
            with st.spinner("🎧 Transcribing speech..."):
                text = transcribe_audio(st.session_state.audio_path)
                st.session_state.transcription = text

            with st.spinner("🧠 Summarizing text..."):
                summary = simple_summarizer(text, num_sentences=3)
                st.session_state.summary = summary

            st.success("✅ Done! Your results are ready below 👇")

        except Exception as e:
            st.error(f"❌ {e}")

    # -------------------- DISPLAY RESULTS --------------------
    if st.session_state.transcription:
        st.markdown("### 📝 Transcribed Text")
        st.text_area("", st.session_state.transcription, height=200)

    if st.session_state.summary:
        st.markdown("### 🧾 Summary")
        st.text_area("", st.session_state.summary, height=150)
        st.download_button("⬇️ Download Transcription", st.session_state.transcription, "transcription.txt")
        st.download_button("⬇️ Download Summary", st.session_state.summary, "summary.txt")

# -------------------- FOOTER --------------------
st.markdown("<hr><p style='text-align:center;color:#94A3B8;'>✨ Built with Streamlit by Prachiti Morankar</p>",
            unsafe_allow_html=True)
