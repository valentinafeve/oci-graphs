import streamlit as st
import io
from pypdf import PdfReader
from graph_pipeline.extract_graph_from_text import extract_graph_from_text, get_conn

st.set_page_config(page_title="GraphRAG on Oracle 23ai", layout="wide")

st.title("Upload Document & Extract Graph")
# st.write("This is the first page of the application.")

# ---------- Helpers: read file content ----------
def read_pdf(file_obj: io.BytesIO) -> str:
    reader = PdfReader(file_obj)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts)

def read_txt(file_obj: io.BytesIO) -> str:
    # Streamlit gives a file-like object with bytes
    return file_obj.read().decode("utf-8", errors="ignore")

def get_text_from_upload(uploaded) -> str:
    name = (uploaded.name or "").lower()
    mime = (uploaded.type or "").lower()

    if name.endswith(".pdf") or "pdf" in mime:
        return read_pdf(uploaded)
    # default to txt-like
    return read_txt(uploaded)

# ---------- UI ----------
uploaded_file = st.file_uploader(
    "Drag & drop PDF or .txt",
    type=["pdf", "txt"],
    accept_multiple_files=False,
)

connection = get_conn()
if connection:
    st.toast("Connected to Oracle Database", icon="✅")

if st.button("Run extraction"):
    if not uploaded_file:
        st.warning("Please upload a file first.")
    else:
        with st.spinner("Extracting graph from text…"):
            text = get_text_from_upload(uploaded_file)
            result = extract_graph_from_text(text, True)
        st.success("✅ Extraction complete!", icon="✅")
        # Optional: show/use the result
        with st.expander("View result"):
            st.json(result)