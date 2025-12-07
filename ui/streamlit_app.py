# place under ui/
import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(layout="wide", page_title="Clip+FAISS + LLaMA Demo")

st.title("clip-faiss-coco — LLaMA caption + explanation demo")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    k = st.slider("Top-k retrieved", 1, 30, 10)
    if st.button("Generate caption"):
        if uploaded is None:
            st.warning("Upload an image first")
        else:
            files = {"file": uploaded.getvalue()}
            # Replace HOST with your API endpoint
            resp = requests.post("http://localhost:8000/generate/caption_from_image", files={"file": uploaded}, data={"k": k})
            if resp.status_code == 200:
                st.success("OK")
                data = resp.json()
                st.write("Title:", data["output"].get("Title"))
                st.write("ShortCaption:", data["output"].get("ShortCaption"))
                st.write("DetailedCaption:", data["output"].get("DetailedCaption"))
                st.write("Tags:", data["output"].get("Tags"))

with col2:
    st.header("Retrieved neighbors (preview)")
    st.write("Top-k will appear here from your FAISS index via the backend")
    st.image([], width=200)
