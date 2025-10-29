import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import time
import base64
import json

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Speech-to-Text Dashboard",
    layout="wide",
    page_icon="🎙️"
)

# ------------------- CUSTOM CSS -------------------
st.markdown("""
<style>
/* ---------- GLOBAL STYLES ---------- */
body, .stApp {
    background-color: #f9fbff;
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

/* ---------- HEADER BAR ---------- */
.header-bar {
    background: linear-gradient(90deg, #60A5FA 0%, #3B82F6 100%);
    color: white;
    padding: 1.2rem 2rem;
    border-radius: 0 0 12px 12px;
    text-align: center;
    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
}
.header-bar h1 {
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.header-bar p {
    font-size: 16px;
    color: #E0F2FE;
}

/* ---------- STATUS BOX ---------- */
.status-container {
    text-align: center;
    margin: 1.5rem 0 2rem 0;
}
.status-box {
    display: inline-block;
    background-color: #E0F2FE;
    color: #1E40AF;
    font-weight: 600;
    padding: 10px 24px;
    border-radius: 12px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.1);
}

/* ---------- CARD DESIGN ---------- */
.card {
    background: white;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    padding: 24px 28px;
    height: 100%;
    transition: all 0.2s ease;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.12);
}

/* ---------- BUTTON STYLING ---------- */
div.stButton > button {
    background: linear-gradient(90deg, #3B82F6, #60A5FA);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #2563EB, #3B82F6);
    transform: scale(1.04);
}

/* ---------- FILE UPLOADER ---------- */
.stFileUploader > div {
    background-color: #F8FAFC;
    border-radius: 12px;
    border: 1px dashed #93C5FD;
    padding: 0.6rem;
}

/* ---------- DOWNLOAD & COPY ---------- */
.controls-right {
    display:flex;
    justify-content:flex-end;
    gap:10px;
    align-items:center;
    margin-top:8px;
}
a.download-link {
    text-decoration:none;
    padding:5px 8px;
    border-radius:6px;
    background:#DBEAFE;
    font-weight:600;
    color:#1D4ED8;
}
button.copy-btn {
    padding:5px 8px;
    border-radius:6px;
    border:1px solid #DBEAFE;
    background:#EFF6FF;
    cursor:pointer;
    font-weight:600;
    color:#1D4ED8;
}

/* ---------- TEXTAREAS ---------- */
textarea {
    border-radius: 10px !important;
    border: 1px solid #E5E7EB !important;
    background-color: #F9FAFB !important;
    color: #1F2937 !important;
}

/* ---------- FOOTER ---------- */
.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 14px;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------- HEADER -------------------
st.markdown("""
<div class="header-bar">
    <h1>🎙️ AI Speech-to-Text Dashboard</h1>
    <p>Record or upload audio → Transcribe, Diarize & Summarize effortlessly</p>
</div>
""", unsafe_allow_html=True)

# ------------------- SESSION STATE -------------------
if "status" not in st.session_state:
    st.session_state.status = "Idle"
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "diarized" not in st.session_state:
    st.session_state.diarized = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "show_webrtc" not in st.session_state:
    st.session_state.show_webrtc = False

# ------------------- HELPERS -------------------
def make_download_and_copy_html(text: str, filename: str) -> str:
    b64 = base64.b64encode(text.encode()).decode()
    download_link = f'<a class="download-link" href="data:file/txt;base64,{b64}" download="{filename}">⬇️</a>'
    js_text = json.dumps(text)
    copy_button = f'<button class="copy-btn" onclick="navigator.clipboard.writeText({js_text})">📋</button>'
    return f'<div class="controls-right">{download_link}{copy_button}</div>'

# ------------------- STATUS -------------------
st.markdown(f"""
<div class="status-container">
    <div class="status-box">🟢 Current Stage: {st.session_state.status}</div>
</div>
""", unsafe_allow_html=True)

# ------------------- MAIN LAYOUT -------------------
col1, col2 = st.columns([1, 2], gap="large")

# ------------------- LEFT: INPUTS -------------------
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🎧 Audio Input")
    input_mode = st.radio("Choose Input Type:", ["🎙️ Live Recording", "📁 Upload Audio File"])

    if input_mode == "🎙️ Live Recording":
        st.write("Click **Start Recording** to begin and **Stop** to process audio.")

        if not st.session_state.show_webrtc:
            if st.button("▶️ Start Recording"):
                st.session_state.show_webrtc = True
                st.session_state.status = "Recording..."

        if st.session_state.show_webrtc:
            webrtc_streamer(
                key="audio",
                mode=WebRtcMode.SENDONLY,
                media_stream_constraints={"audio": True, "video": False},
                async_processing=False,
            )

            if st.button("⏹️ Stop Recording"):
                st.session_state.show_webrtc = False
                st.session_state.status = "Processing (Transcribing)..."
                time.sleep(1)
                st.session_state.transcription = "This is a sample transcription from your recorded audio."
                st.session_state.status = "Processing (Diarizing)..."
                time.sleep(1)
                st.session_state.diarized = "Speaker 1: Hello!\nSpeaker 2: Hi there!"
                st.session_state.status = "Processing (Summarizing)..."
                time.sleep(1)
                st.session_state.summary = "Summary: Two speakers greeted each other briefly."
                st.session_state.status = "✅ Completed"

    elif input_mode == "📁 Upload Audio File":
        uploaded_audio = st.file_uploader("Upload an audio file (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])
        if uploaded_audio:
            st.audio(uploaded_audio)
            st.success("✅ Audio uploaded successfully!")
            if st.button("🚀 Process Audio"):
                st.session_state.status = "Processing (Transcribing)..."
                time.sleep(1)
                st.session_state.transcription = "This is a sample transcription from your uploaded audio file."
                st.session_state.diarized = "Speaker 1: Hey\nSpeaker 2: Hi"
                st.session_state.summary = "Summary: Short greeting exchange."
                st.session_state.status = "✅ Completed"

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------- RIGHT: OUTPUT -------------------
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🧾 Processed Output")

    tab1, tab2, tab3 = st.tabs(["📝 Transcription", "👥 Diarized Text", "🧠 Summary"])

    with tab1:
        st.subheader("Raw Transcription")
        st.text_area("Transcribed Text", value=st.session_state.transcription, height=250, key="ta_transcription")
        st.markdown(make_download_and_copy_html(st.session_state.transcription, "transcription.txt"), unsafe_allow_html=True)

    with tab2:
        st.subheader("Speaker-Diarized Output")
        st.text_area("Diarized Output", value=st.session_state.diarized, height=250, key="ta_diarized")
        st.markdown(make_download_and_copy_html(st.session_state.diarized, "diarized_output.txt"), unsafe_allow_html=True)

    with tab3:
        st.subheader("Meeting Summary / Notes")
        st.text_area("Summarized Notes", value=st.session_state.summary, height=250, key="ta_summary")
        st.markdown(make_download_and_copy_html(st.session_state.summary, "summary.txt"), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------- FOOTER -------------------
st.markdown("<hr><p class='footer'>Built using Streamlit</p>", unsafe_allow_html=True)
