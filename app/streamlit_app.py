"""
Streamlit prototype UI for the Airport Multimodal Passenger Assistance Chatbot.

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
from ui_helpers import confidence_color, format_record_card, EXAMPLE_QUERIES, PRIVACY_NOTICE

st.set_page_config(page_title="Airport Assistant", page_icon="🛫", layout="centered")

st.title("🛫 Airport Passenger Assistance Chatbot")
st.caption("Proof-of-concept multimodal assistant -- text, voice, and image input")

with st.expander("ℹ️ Privacy notice", expanded=False):
    st.warning(PRIVACY_NOTICE)

st.subheader("Ask a question")

col1, col2 = st.columns(2)
with col1:
    text_input = st.text_input("Type your question", placeholder="e.g. Where is gate B12?")
with col2:
    st.write("Or try an example:")
    example = st.selectbox("Example queries", ["-- select --"] + EXAMPLE_QUERIES, label_visibility="collapsed")
    if example != "-- select --":
        text_input = example

st.subheader("Optional inputs")
audio_file = st.file_uploader("Upload a voice query (.wav/.mp3)", type=["wav", "mp3", "m4a"])
image_file = st.file_uploader("Upload an airport sign/photo", type=["png", "jpg", "jpeg"])

image_obj = None
if image_file is not None:
    image_obj = Image.open(image_file).convert("RGB")
    st.image(image_obj, caption="Uploaded image", width=250)

audio_path = None
if audio_file is not None:
    tmp_path = os.path.join("outputs", "predictions", f"_tmp_{audio_file.name}")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(audio_file.getbuffer())
    audio_path = tmp_path
    st.audio(audio_file)

if st.button("Get answer", type="primary"):
    if not text_input and image_obj is None and audio_path is None:
        st.error("Please provide at least one input: text, voice, or image.")
    else:
        with st.spinner("Processing..."):
            result = run_inference(
                text=text_input if text_input else None,
                image=image_obj,
                audio_path=audio_path,
            )

        st.subheader("Response")

        transcript_text = result["input_summary"]["transcript"]
        if transcript_text:
            st.info(f"🎤 Transcribed voice query: \"{transcript_text}\"")

        conf = result["confidence"]
        color = confidence_color(conf)
        st.markdown(f"**Confidence:** :{color}[{conf:.2f}]")

        if result["uncertain"]:
            st.error(result["message"])
        else:
            st.success(result["message"])

        with st.expander("Matched knowledge base record"):
            st.markdown(format_record_card(result["matched_record"]))

        with st.expander("Rationale / how this answer was produced"):
            st.write(result["rationale"])

st.divider()
st.caption(
    "This is an academic proof-of-concept. Information may be outdated or incorrect. "
    "For real travel decisions, always confirm with airport staff or official displays."
)
