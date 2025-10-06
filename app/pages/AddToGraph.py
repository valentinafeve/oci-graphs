import streamlit as st
from graph_pipeline.extract_graph_from_text import extract_graph_from_text

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Insert to knowledge Graph", layout="wide")
st.sidebar.caption("Oracle Database 23ai • Property Graph • Oracle GenAI")

st.title("GraphRAG: Update Knowledge Graph")

input = st.text_input("Insert a sentence into the knowledge graph:", placeholder="e.g., Albert Einstein liked pizza")

if st.button("Run", disabled=not input):
    with st.spinner("Thinking..."):
        extract_graph_from_text(input, False)
        st.success("✅ Update complete! This statement will now be reflected in the knowledge graph", icon="✅")