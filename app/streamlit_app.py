"""
Streamlit prototype UI for the Airport Multimodal Passenger Assistance Chatbot.

Unified single-bar chat interface: one text box + one attach button that
accepts EITHER an image OR an audio file (or none). Text, image, and audio
can all be combined in a single turn and are routed through the same
multimodal fusion engine.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
from PIL import Image

from inference import run_inference
from ui_helpers import (
    confidence_color,
    confidence_bar_html,
    format_record_card,
    EXAMPLE_QUERIES,
    PRIVACY_NOTICE,
)

st.set_page_config(
    page_title="Airport Assistant",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global CSS: theme, chat bubbles, input bar, progress bar, clear button
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Page background & base font ───────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0f1117;
        color: #e8eaf0;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }

    /* ── App title ──────────────────────────────────────────── */
    h1 {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #e8eaf0 !important;
        letter-spacing: -0.01em;
    }

    /* ── Chat bubbles ───────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 12px 16px;
        margin-bottom: 6px;
        max-width: 88%;
    }
    /* user bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #1e3a5f;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    /* assistant bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: #1a1d27;
        border: 1px solid rgba(255,255,255,0.07);
        border-bottom-left-radius: 4px;
    }

    /* ── Input chat bar ─────────────────────────────────────── */
    .chatbar-wrap {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 28px;
        padding: 6px 14px;
        background: #1a1d27;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .chatbar-wrap input {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        color: #e8eaf0 !important;
    }

    /* ── Send button ────────────────────────────────────────── */
    button[kind="primary"] {
        background: linear-gradient(135deg, #1565c0, #1e88e5) !important;
        border: none !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        color: #fff !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1976d2, #42a5f5) !important;
    }

    /* ── Clear chat button ──────────────────────────────────── */
    .clear-btn button {
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 16px !important;
        color: #888 !important;
        font-size: 0.75rem !important;
        padding: 2px 12px !important;
    }
    .clear-btn button:hover {
        border-color: #f44336 !important;
        color: #f44336 !important;
    }

    /* ── Attach chip ────────────────────────────────────────── */
    .attach-chip {
        font-size: 0.78rem;
        color: #64b5f6;
        margin: 5px 0 0 6px;
    }

    /* ── Expander headers ───────────────────────────────────── */
    details summary {
        font-size: 0.82rem !important;
        color: #90a4ae !important;
    }

    /* ── Divider ─────────────────────────────────────────────── */
    hr { border-color: rgba(255,255,255,0.07) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header row: title + clear chat button
# ---------------------------------------------------------------------------
title_col, clear_col = st.columns([6, 1])
with title_col:
    st.title("✈️ Airport Assistant")
    st.caption(
        "Ask by typing, and optionally attach a photo or voice clip — "
        "or use your camera or mic — all from a single bar below."
    )
with clear_col:
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("🗑 Clear", key="clear_chat"):
        st.session_state.history = []
        st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("ℹ️ Privacy notice", expanded=False):
    st.warning(PRIVACY_NOTICE)

with st.expander("💡 Example questions", expanded=False):
    for q in EXAMPLE_QUERIES:
        st.markdown(f"- {q}")

# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"], unsafe_allow_html=True)
        extra = turn.get("extra")
        if extra:
            if extra.get("image_bytes"):
                st.image(extra["image_bytes"], width=180)
            if extra.get("transcript"):
                st.caption(f'🎤 Transcribed: "{extra["transcript"]}"')
            if extra.get("record"):
                with st.expander("📋 Matched knowledge base record"):
                    st.markdown(format_record_card(extra["record"]))
            if extra.get("rationale"):
                with st.expander("🔍 How this answer was produced"):
                    st.write(extra["rationale"])

# ---------------------------------------------------------------------------
# Single unified input bar: attach + text + send
# ---------------------------------------------------------------------------
st.markdown('<div class="chatbar-wrap">', unsafe_allow_html=True)
col_attach, col_text, col_send = st.columns([0.8, 7.2, 1.2])

attached_files = None
camera_photo = None
mic_recording = None

with col_attach:
    with st.popover("📎"):
        tab_upload, tab_cam, tab_mic = st.tabs(["Upload", "Camera", "Mic"])

        with tab_upload:
            attached_files = st.file_uploader(
                "Attach image or audio",
                type=["jpg", "jpeg", "png", "webp", "wav", "mp3", "m4a", "ogg"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key=f"uploader_{st.session_state.uploader_key}",
            )

        with tab_cam:
            camera_photo = st.camera_input(
                "Take a photo",
                label_visibility="collapsed",
                key=f"cam_{st.session_state.uploader_key}",
            )

        with tab_mic:
            if hasattr(st, "audio_input"):
                mic_recording = st.audio_input(
                    "Record a voice question",
                    label_visibility="collapsed",
                    key=f"mic_{st.session_state.uploader_key}",
                )
            else:
                mic_recording = None
                st.info(
                    "Live mic recording needs Streamlit >= 1.36. "
                    "Use the Upload tab to attach a .wav/.mp3/.m4a file instead."
                )

    # Merge sources into at most one image + one audio
    attached_image_file = None
    attached_audio_file = None

    if attached_files:
        for f in attached_files:
            if f.type.startswith("image"):
                if attached_image_file is None:
                    attached_image_file = f
                else:
                    st.warning(f"Only one image supported — ignoring '{f.name}'.")
            else:
                if attached_audio_file is None:
                    attached_audio_file = f
                else:
                    st.warning(f"Only one audio clip supported — ignoring '{f.name}'.")

    if camera_photo is not None:
        if attached_image_file is None:
            attached_image_file = camera_photo
        else:
            st.warning("Image already uploaded — ignoring camera photo.")

    if mic_recording is not None:
        if attached_audio_file is None:
            attached_audio_file = mic_recording
        else:
            st.warning("Audio already uploaded — ignoring mic recording.")

with col_text:
    text_input = st.text_input(
        "Message",
        placeholder="Ask something, e.g. Where is gate B12?",
        label_visibility="collapsed",
        key=f"text_{st.session_state.uploader_key}",
    )

with col_send:
    send_clicked = st.button("Send ➤", type="primary", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

attach_chips = []
if attached_image_file is not None:
    attach_chips.append(f"🖼️ {attached_image_file.name}")
if attached_audio_file is not None:
    attach_chips.append(f"🎤 {attached_audio_file.name}")
if attach_chips:
    st.markdown(
        f'<div class="attach-chip">📎 Attached: {", ".join(attach_chips)}</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Handle send
# ---------------------------------------------------------------------------
if send_clicked:
    if not text_input and attached_image_file is None and attached_audio_file is None:
        st.error("Please type a question or attach an image / voice clip.")
    else:
        image_obj = None
        audio_path = None
        image_bytes_for_display = None

        if attached_image_file is not None:
            image_obj = Image.open(attached_image_file).convert("RGB")
            image_bytes_for_display = attached_image_file.getvalue()

        if attached_audio_file is not None:
            tmp_path = os.path.join("outputs", "predictions", f"_tmp_{attached_audio_file.name}")
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(attached_audio_file.getbuffer())
            audio_path = tmp_path

        user_display = text_input if text_input else "_(attachment only)_"
        st.session_state.history.append({
            "role": "user",
            "content": user_display,
            "extra": {"image_bytes": image_bytes_for_display} if image_bytes_for_display else None,
        })

        with st.spinner("Thinking..."):
            result = run_inference(
                text=text_input if text_input else None,
                image=image_obj,
                audio_path=audio_path,
            )

        conf = result["confidence"]
        transcript_text = result["input_summary"]["transcript"]

        # Build response: progress bar + message text
        response_md = confidence_bar_html(conf) + "\n\n" + result["message"]

        st.session_state.history.append({
            "role": "assistant",
            "content": response_md,
            "extra": {
                "transcript": transcript_text,
                "record": result["matched_record"],
                "rationale": result["rationale"],
            },
        })

        st.session_state.uploader_key += 1
        st.rerun()

st.divider()
st.caption(
    "🎓 Academic proof-of-concept. Always verify flight information with "
    "official airport displays or staff."
)
