"""
Streamlit demo for CLIP + FAISS retrieval

Usage:
    cd clip-faiss-coco
    pip install streamlit requests pillow
    streamlit run ui/streamlit_app.py
"""
import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
import os
from typing import List

# Config
API_URL = st.sidebar.text_input("API URL", "http://localhost:8000")
DEFAULT_K = st.sidebar.slider("Top-k",1,20,6)
st.title("CLIP + FAISS Retrieval Demo")
st.markdown("Type text (-> retrieve images) or upload an image (-> retrieve similar images)")

# Helpers
def call_text_search(query:str, k:int=6):
    url = f"{API_URL}/search/text"
    payload = {"query":query, "k":k}
    try:
        resp = requests.post(url, json=payload,timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(e)
        return None

def call_image_search_bytes(image_bytes:bytes,filename:str='query.jpg',k:int=6):
    url = f"{API_URL}/search/image"
    files = {'file':(filename, image_bytes,"image/jpeg")}
    try:
        resp = requests.post(url, files=files, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Image search API error: {e}")
        return None
def show_results_grid(results:List[dict],cols:int=3):
    if not results:
        st.info("No results display")
        return
    # Determine grid columns
    rows = (len(results) + cols - 1) // cols
    i = 0
    for r in range(rows):
        cols_ui = st.columns(cols)
        for c in range(cols):
            if i >= len(results):
                break
            res = results[i]
            with cols_ui[c]:
                st.markdown(f"**Rank {res.get('rank',i+1)}**")
                # Use 'image_path' if accessible locally; otherwise try to fetch via API (not implemented)
                image_path = res.get('image_path')
                caption = res.get('caption','')
                score = res.get('score',None)

                shown=False
                if image_path:
                    # If path exists on this machine, open and show
                    if os.path.exists(image_path):
                        try:
                            img = Image.open(image_path).convert('RGB')
                            st.image(img, use_column_width='always')
                            shown=True
                        except Exception as e:
                            shown=False

                if not shown:
                    st.write("(image not available locally)")
                    # optionally display a placeholder box
                    st.empty()

                st.caption(f"{caption}\n\n score: {score}")
            i += 1

# ----------- UI : Text Search --------
st.header("Text search")
with st.form("text_search_form",clear_on_submit=False):
    query = st.text_input("Search text (query)",value="a dog on a skateboard")
    k_text = st.number_input("Top-k",min_value=1,max_value=50, value=DEFAULT_K,step=1)
    submit_text = st.form_submit_button("Search (text)")
if submit_text:
    if not query or query.strip() == "":
        st.warning("Please enter a non-empty query")
    else:
        with st.spinner("Searching..."):
            js = call_text_search(query.strip(),int(k_text))
        if js:
            st.success(f"Found {len(js.get('results',[]))} results")
            show_results_grid(js.get("results",[]),cols=3)
st.markdown("----")

# UI: Image Search
st.header("Image search (upload)")
uploaded_file = st.file_uploader("Upload an image",type=["jpg","jpeg","png"])
k_img = st.number_input("Top-k",min_value=1,max_value=50,value=DEFAULT_K,step=1,key="k_img")
if uploaded_file is not None:
    img_bytes = uploaded_file.read()
    # previes
    try:
        img_preview = Image.open(BytesIO(img_bytes))
        st.image(img_preview, caption="Query image",use_column_width=True)
    except Exception as e:
        st.warning("Could not display uploaded image preview")

    if st.button("Search (image)"):
        with st.spinner("Searching..."):
            js = call_image_search_bytes(img_bytes,uploaded_file.name,k=int(k_img))
        if js:
            st.success(f"Found {len(js.get('results',[]))} results")
            show_results_grid(js.get("results",[]),cols=3)
st.sidebar.markdown("---")
st.sidebar.markdown("Developer notes: ")
st.sidebar.markdown("- Streamlit runs locally and expects access to the same dataset files referenced in metadata (image_path).")
st.sidebar.markdown("- If images are not available locally, extend the backend to serve images (e.g., FastAPI static files) or modify API or return image bytes/urls.")