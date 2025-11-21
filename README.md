🎙️ AI Live Meeting Summarizer

Real-time Audio Recording → Speech-to-Text → Automatic Summaries → Export & Email Support

📌 Overview

The AI Live Meeting Summarizer is a Streamlit-based application that allows users to:

✔️ Record live audio
✔️ Upload .wav audio
✔️ Convert speech to text (Google Speech Recognition)
✔️ Generate automatic summaries (TF-IDF Extractive Summarizer)
✔️ Download outputs as .txt, .md, .pdf
✔️ Email results with attachments
✔️ Store structured logs (.json / .csv)
✔️ No API Keys required
✔️ No heavy ML models or C++ dependencies

This tool is ideal for:

Meeting transcription

Classroom lectures

Interviews

Online calls

Content summarization

🚀 Features
🎤 1. Speech-to-Text

Live audio recording using sounddevice

Upload .wav files

Transcription powered by Google Web Speech API (free)

Works without API keys

🧠 2. Smart Text Summarizer

Uses TF-IDF Extractive Summarization (sklearn) to generate meaningful summaries.

✔ Identifies important sentences
✔ Lightweight — works offline
✔ No GPU/ML model required

🎨 3. Beautiful UI (Streamlit)

Modern UI with gradients

Clean layout (input left, output right)

Audio playback

Replay / Reset buttons

Responsive design

📤 4. Export Options

Export your results in multiple formats:

Format	Contents Included
.txt	Transcript / Summary
.md	Structured meeting summary
.pdf	Printable summary with headings

PDFs use a Unicode-safe font to prevent emoji errors.

✉️ 5. Email Integration

Send transcript + summary via email with one click.

Supports:

Gmail (App Password required)

Outlook / Yahoo / Custom SMTP

Attachments included:

meeting_summary.md

meeting_summary.pdf

🗂️ 6. Structured Logging

Each processed session is automatically saved as:

session.json

session.csv

Containing:

Field	Description
Title	Meeting title
Date	Meeting date
Speakers	Speaker names
Transcript	Full text
Summary	Auto summary
Timestamp	Unique session ID

🏗️ Architecture Overview
Audio Input (Record/Upload)
        ↓
Speech Recognition (Google Web Speech API)
        ↓
Transcription Text
        ↓
TF-IDF Extractive Summarizer
        ↓
Summary Generated
        ↓
Export (.txt/.md/.pdf) + Email + Logging

🧰 Tech Stack
Frontend

Streamlit — UI, layout, audio playback

Audio Processing

sounddevice — live recording

wave — store WAV files

SpeechRecognition — transcription (Google)

Summarization

sklearn.feature_extraction.text.TfidfVectorizer

numpy

regex

Export & Email

fpdf — PDF generation

smtplib — email system

email.mime — attachments handling

Logging

json

csv

(Optional) pyarrow for .parquet

📁 Project Structure
📦 AI Live Meeting Summarizer
│
├── Milestone 1 – Basic STT
├── Milestone 2 – Diarization & Summarization
├── Milestone 3 – UI + Full Integration
├── Milestone 4 – Email + Export + Logging + Final App
│
├── App.py  ← Main Application
├── README.md
└── assets/

🧪 Milestones Summary
Milestone 1 – Speech-to-Text (WER Evaluation)

Implemented STT using Vosk/Whisper

Evaluated using jiwer

Achieved WER < 15%

Milestone 2 – Diarization + Summarization

Explored pyannote diarization (Torch-based, optional)

Implemented lightweight TF-IDF summary

ROUGE > 0.4

Milestone 3 – UI Integration

Full Streamlit UI

File upload + recording

Display sections

No UI lag or errors

Milestone 4 – Final System

Export to .md, .pdf

Email system with attachments

Structured logging

Fully functional pipeline

🧩 Challenges I Faced

✔ Issues with Whisper/Torch DLL errors
✔ Unicode/emoji errors in PDF export
✔ Gmail rejecting passwords (fixed via App Passwords)
✔ Git submodule accidental uploads
✔ Large PPT files not previewing on GitHub
✔ Fixing MD/PDF export showing empty content
✔ Merging updated Milestone 3 & 4 folders without conflicts

🔮 Future Enhancements

🔥 Add real machine-learning summarizers:

BART

T5

LLaMA 3.1 via Groq API

🎙️ Add real-time streaming STT
🧑‍🤝‍🧑 Add speaker diarization using pyannote
🗂️ Add database (PostgreSQL or Firebase) for history
🌐 Deploy on cloud (Streamlit Cloud / Render)
📊 Add analytics dashboard
📱 Make mobile-friendly interface

▶️ Demo Flow

Open app

Choose Record Audio OR Upload File

Click Process Audio

View transcription

View summary

Download or send via email

Logs saved automatically

💻 Installation
1️⃣ Clone the repository
git clone https://github.com/Speech-Summarizer-Application-IF-SB/Prachiti_morankar.git
cd Prachiti_morankar/Milestone\ 

2️⃣ Install requirements
pip install -r requirements.txt

3️⃣ Run app
streamlit run App.py
