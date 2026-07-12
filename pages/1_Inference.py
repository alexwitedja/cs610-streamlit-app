import streamlit as st

st.set_page_config(
    page_title="Inference",
    page_icon="",
)

st.title("Inference")

st.sidebar.header("Inference")
mode = st.sidebar.radio(
    "Choose model:",
    ["Patient", "Hospital"],
)

if mode == "Patient":
    pass
else:
    pass