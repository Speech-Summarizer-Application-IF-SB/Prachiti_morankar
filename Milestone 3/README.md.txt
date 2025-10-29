# 🎙️ AI Speech-to-Text Dashboard

A clean, modern Streamlit web app that records or uploads audio and converts it to text — complete with speaker diarization and meeting summary features.

## 🚀 Features
- 🎧 Record live audio directly in your browser  
- 📁 Upload `.wav`, `.mp3`, or `.m4a` audio files  
- 📝 Transcription view  
- 👥 Speaker Diarization  
- 🧠 Automatic Summary Generation  
- 📋 Copy & Download text buttons  
- ✨ Modern light theme with soft shadows and gradients  

## 🧩 Tech Stack
- **Frontend/UI:** Streamlit  
- **Audio Processing:** pydub, soundfile  
- **Backend (optional):** OpenAI Whisper or any ASR model  

## ⚙️ Installation
```bash
git clone https://github.com/YOUR-USERNAME/SpeechToText-App.git
cd SpeechToText-App
pip install -r requirements.txt
streamlit run App.py
