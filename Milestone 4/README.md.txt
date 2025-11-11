# 🎙️ Speech-to-Text + Summarizer (Milestone 4 – BuildSmart)

## 📘 Overview
This project is a **Streamlit-based AI dashboard** that converts speech to text, summarizes the transcript, and exports or emails the results.  
It supports **live recording** and **audio file upload**, works **without any API key**, and includes export, email, and structured logging.

---

## 🧩 Core Modules

| Module | Description |
|---------|-------------|
| **Speech Recognition (STT)** | Converts voice to text using Google’s free Web Speech API. |
| **Summarization** | Extractive text summarizer using TF-IDF ranking. |
| **Export** | Exports meeting transcript and summary to `.md` and `.pdf`. |
| **Email System** | Sends meeting summary and transcript as attachments via SMTP. |
| **Structured Logging** | Saves meeting sessions with timestamp, metadata, and content. |

---

## ⚙️ Architecture

```text
🎙️ Audio Input (Mic / File)
        ↓
SpeechRecognition (Google Web Speech)
        ↓
TF-IDF Summarizer (Scikit-learn + NumPy)
        ↓
🧾 Output (Transcript + Summary)
        ↓
📄 Export (.md / .pdf)     ✉️ Email Sender
        ↓
🗂️ Structured Logging (.json / .parquet)
