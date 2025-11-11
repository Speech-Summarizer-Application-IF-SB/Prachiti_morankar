import os, re, json, wave, socket, tempfile, traceback
from datetime import datetime
import streamlit as st
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional dependency for PDF
try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="🎙️ AI Live Meeting Summarizer", layout="wide")

# -------------------- STYLE --------------------
st.markdown("""
<style>
body, .stApp { background-color: #f7faff; font-family: 'Inter', sans-serif; }
h1, h2, h3 { color: #1E3A8A; }
textarea { border-radius: 10px !important; border: 1px solid #CBD5E1 !important; background: #F9FAFB !important; }
button[kind="primary"] {
    background: linear-gradient(90deg, #2563EB, #60A5FA) !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
button[kind="primary"]:hover { transform: scale(1.03); }
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div style='background: linear-gradient(90deg,#3B82F6,#60A5FA); padding:1rem 2rem; 
border-radius:0 0 14px 14px; text-align:center; color:white;'>
  <h1>🎙️ AI Live Meeting Summarizer</h1>
  <p>Record or upload audio → Transcribe → Summarize → Export / Email</p>
</div>
""", unsafe_allow_html=True)

# -------------------- SESSION DEFAULTS --------------------
if "audio_path" not in st.session_state: st.session_state.audio_path = None
if "transcription" not in st.session_state: st.session_state.transcription = ""
if "summary" not in st.session_state: st.session_state.summary = ""
if "meta" not in st.session_state: st.session_state.meta = {}
if "email_cfg" not in st.session_state:
    st.session_state.email_cfg = {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": "465",
        "email_user": "",
        "email_pass": "",
        "email_to": "",
        "subject": ""
    }

recognizer = sr.Recognizer()

# -------------------- HELPERS --------------------
def internet_ok(host="8.8.8.8", port=53, timeout=3):
    """Check internet connection."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False

def summarize_tfidf(text, num_sentences=3):
    """Summarize text using TF-IDF."""
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= num_sentences: return text
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()
    idx = np.argsort(scores)[::-1][:num_sentences]
    return " ".join([sentences[i] for i in sorted(idx)])

def transcribe_google(path):
    """Transcribe audio using Google Speech."""
    with sr.AudioFile(path) as src:
        audio = recognizer.record(src)
    return recognizer.recognize_google(audio)

def record_audio(duration=5, fs=44100):
    """Record mic audio to .wav."""
    st.info(f"🎤 Recording for {duration} seconds... Speak now!")
    rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp.name, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(fs)
        wf.writeframes(rec.tobytes())
    st.success("✅ Recording complete! Click 'Process Audio'.")
    return tmp.name

def build_markdown(title, date, transcript, summary, speakers=""):
    """Create markdown export."""
    return f"""# {title or 'Meeting Summary'}
_Date_: {date}{f"  |  _Speakers_: {speakers}" if speakers else ""}

---

## Transcription
{transcript or "_(empty)_"}

## Summary
{summary or "_(empty)_"}
"""

# -------------------- PDF EXPORT (Unicode Safe) --------------------
def md_to_pdf_bytes(md_text):
    """Convert markdown to PDF (Unicode + Emoji supported)."""
    from fpdf import FPDF
    import urllib.request
    import os

    font_path = os.path.join(tempfile.gettempdir(), "DejaVuSans.ttf")
    FONT_URLS = [
        "https://github.com/dejavu-fonts/dejavu-fonts/raw/version_2_37/ttf/DejaVuSans.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/apache/dejavu/DejaVuSans.ttf"
    ]
    if not os.path.exists(font_path):
        for url in FONT_URLS:
            try:
                urllib.request.urlretrieve(url, font_path)
                break
            except Exception:
                continue
        if not os.path.exists(font_path):
            raise RuntimeError("⚠️ Font download failed. Please check your internet.")

    pdf = FPDF()
    pdf.set_left_margin(12)
    pdf.set_right_margin(12)
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", size=12)

    width = pdf.w - pdf.l_margin - pdf.r_margin
    safe_text = md_text.encode("utf-8", "ignore").decode("utf-8")

    for line in safe_text.split("\n"):
        pdf.multi_cell(width, 6, line)

    output = pdf.output(dest="S")
    if isinstance(output, bytearray):
        return bytes(output)
    elif isinstance(output, str):
        return output.encode("latin1", "ignore")
    else:
        return output

# -------------------- EMAIL SYSTEM --------------------
def send_email(smtp_host, smtp_port, user, pwd, to, subject, body, attachments):
    """Send email safely with UTF-8 headers."""
    import re, smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    from email.header import Header
    from email.utils import formataddr

    smtp_host = smtp_host.strip()
    user = user.strip()
    to = to.strip()
    subject = subject.strip()
    body = body.strip()

    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", to):
        raise ValueError(f"Invalid recipient email: '{to}'")

    msg = MIMEMultipart()
    msg["From"] = formataddr((str(Header("Speech-to-Text App", "utf-8")), user))
    msg["To"] = to
    msg["Subject"] = Header(subject or "Meeting Summary", "utf-8")
    msg.attach(MIMEText(body, "plain", "utf-8"))

    for fname, data, main, sub in attachments:
        part = MIMEBase(main, sub)
        part.set_payload(data)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{fname}"')
        msg.attach(part)

    with smtplib.SMTP_SSL(smtp_host, int(smtp_port)) as server:
        server.login(user, pwd)
        server.sendmail(user, [to], msg.as_string())

# -------------------- SIDEBAR --------------------
st.sidebar.header("🧾 Session Details")
title = st.sidebar.text_input("Title", "Meeting Summary")
date_str = st.sidebar.text_input("Date", datetime.now().strftime("%Y-%m-%d"))
speakers = st.sidebar.text_input("Speakers (optional)", "")

st.sidebar.markdown("---")
st.sidebar.subheader("📧 Email / Export")

cfg = st.session_state.email_cfg
cfg["smtp_host"] = st.sidebar.text_input("SMTP Host", cfg["smtp_host"])
cfg["smtp_port"] = st.sidebar.text_input("SMTP Port", cfg["smtp_port"])
cfg["email_user"] = st.sidebar.text_input("From Email", cfg["email_user"])
cfg["email_pass"] = st.sidebar.text_input("App Password", type="password", value=cfg["email_pass"])
cfg["email_to"] = st.sidebar.text_input("To Email", cfg["email_to"])
cfg["subject"] = st.sidebar.text_input("Subject", cfg["subject"] or f"Meeting Summary – {date_str}")

btn_test_email = st.sidebar.button("📧 Test Email Connection")
btn_send_email = st.sidebar.button("✉️ Send Email")

# -------------------- MAIN UI --------------------
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("🎧 Input Options")
    mode = st.radio("Choose Input:", ["🎙️ Record Live", "📁 Upload WAV"])
    if mode == "🎙️ Record Live":
        dur = st.slider("Duration (s)", 3, 20, 5)
        if st.button("🎤 Start Recording"):
            st.session_state.audio_path = record_audio(dur)
            st.audio(st.session_state.audio_path)
    else:
        up = st.file_uploader("Upload a .wav file", type=["wav"])
        if up:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(up.read())
            st.session_state.audio_path = tmp.name
            st.audio(tmp.name)
            st.success("✅ File uploaded!")

with col2:
    st.subheader("🧠 Process & Output")
    if st.session_state.audio_path and st.button("🚀 Process Audio"):
        try:
            with st.spinner("🎧 Transcribing..."):
                text = transcribe_google(st.session_state.audio_path)
            with st.spinner("🧠 Summarizing..."):
                summary = summarize_tfidf(text)
            st.session_state.transcription = text
            st.session_state.summary = summary
            st.success("✅ Done! See below.")
        except Exception as e:
            st.error(f"❌ {e}")
            st.code(traceback.format_exc())

    if st.session_state.transcription:
        st.markdown("### 📝 Transcription")
        st.text_area("", st.session_state.transcription, height=200)
    if st.session_state.summary:
        st.markdown("### 🧾 Summary")
        st.text_area("", st.session_state.summary, height=150)

        # ✅ Build Markdown from session state
        md = build_markdown(title, date_str, st.session_state.transcription or "", st.session_state.summary or "", speakers)

        # ✅ Export buttons
        st.download_button("⬇️ Download Markdown (.md)", data=md.encode("utf-8"), file_name=f"{title or 'summary'}.md", mime="text/markdown")
        if HAS_FPDF:
            pdf_bytes = md_to_pdf_bytes(md)
            st.download_button("📄 Download PDF (.pdf)", data=pdf_bytes, file_name=f"{title or 'summary'}.pdf", mime="application/pdf")

        # ✅ Replay & Clear
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🎧 Replay Last Recording") and st.session_state.audio_path:
                st.audio(st.session_state.audio_path)
        with c2:
            if st.button("🗑️ Clear All"):
                for k in ["audio_path", "transcription", "summary"]:
                    st.session_state[k] = None if k == "audio_path" else ""
                st.experimental_rerun()

# -------------------- EMAIL HANDLERS --------------------
if btn_test_email:
    try:
        send_email(cfg["smtp_host"], cfg["smtp_port"], cfg["email_user"], cfg["email_pass"],
                   cfg["email_to"], "Test Email", "This is a test email from Streamlit.", [])
        st.sidebar.success("✅ Test email sent successfully!")
    except Exception as e:
        st.sidebar.error(f"Email test failed: {e}")

if btn_send_email:
    try:
        if not (st.session_state.transcription or st.session_state.summary):
            st.sidebar.error("⚠️ Please process audio first before sending email.")
        else:
            md = build_markdown(title, date_str, st.session_state.transcription or "", st.session_state.summary or "", speakers)
            attachments = [(f"{title or 'summary'}.md", md.encode("utf-8"), "text", "markdown")]
            if HAS_FPDF:
                pdf_bytes = md_to_pdf_bytes(md)
                attachments.append((f"{title or 'summary'}.pdf", pdf_bytes, "application", "pdf"))

            body = f"Here is your meeting summary for {date_str}.\n\nTitle: {title}\nSpeakers: {speakers}\n\n— Sent from Streamlit App"
            send_email(cfg["smtp_host"], cfg["smtp_port"], cfg["email_user"], cfg["email_pass"],
                       cfg["email_to"], cfg["subject"] or f"Meeting Summary – {date_str}", body, attachments)
            st.sidebar.success(f"✅ Email sent successfully to {cfg['email_to']}!")
    except Exception as e:
        st.sidebar.error(f"Email failed: {e}")
        st.sidebar.code(traceback.format_exc(), language="text")

st.markdown("<hr><p style='text-align:center;color:#94A3B8;'>✨ Built with Streamlit by Prachiti Morankar </p>", unsafe_allow_html=True)
