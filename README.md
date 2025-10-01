# 🎙 Meeting Summarizer Application

A web application that **uploads or records meeting audio, removes background noise, and transcribes it to text** using state-of-the-art speech recognition and noise reduction.

---

**Features**

- Upload or record audio (wav/mp3) of meetings.
- Automatic **noise reduction** for clearer transcription.
- **Transcription** powered by OpenAI's Whisper ASR model.
- Play back cleaned audio.
- Lightweight and fast, works on CPU.

---

**Demo**

You can try the live demo on Hugging Face Spaces:  
(https://huggingface.co/spaces/Prachumm/meeting-summarizer)


**Technologies Used** :- 

-Python 3.10+
-Gradio
 – > Web UI for audio upload/recording
-noisereduce
 – > Noise reduction
-Transformers
 – > Whisper ASR model
-SciPy
 – > Audio resampling


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
