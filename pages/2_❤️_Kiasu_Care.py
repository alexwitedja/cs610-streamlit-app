import streamlit as st
from groq import Groq
from pathlib import Path

PROMPTS_DIR = Path().resolve() / "artifacts" / "prompts"

st.title("Kiasu Care ❤️")

with st.expander("ℹ️ Disclaimer"):
    st.write(
        """
        KakiCare is a CS610 student project built on MIMIC-IV-ED, a de-identified dataset from a **US** hospital
        system. It has not been validated on Singapore patients, has not been reviewed by any clinician, and is
        **not a medical device**. Nothing here is medical advice. In an emergency, call **995**.
        """
    )

st.sidebar.warning("""
⚠️ **A model trained in America is giving you advice about Singapore.**

Read the number as **"roughly how urgent does this pattern look?"** — not as a triage category any
Singapore hospital would recognise.
""")

with st.sidebar.expander("Why this matters?"):
    st.write(
        "1. Singapore uses PACS not ESI, we have not validated this translation to Singapore triage system.\n\n"
        "2. The American health care system is different to Singapore. In America, it is common for ED to be the first stop while Singapore has a dense GP & polyclinic layer long before patient reaches A&E.\n\n"
        "3. Clinical safety thresholds are not guaranteed to be the same for Singaporean population.\n\n"
        "4. Advice layer is not based on data and needs clinician and local validation."
    )

# Set OpenAI API key from Streamlit secrets
@st.cache_resource
def init_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

client = init_groq_client()

# Set a default model
if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = "llama-3.3-70b-versatile"

# Initialize chat history
if "messages" not in st.session_state:
    with open(PROMPTS_DIR / "system_prompt.md") as f:
        system_prompt = f.read()

    with open(PROMPTS_DIR / "assistant_prompt.md") as f:
        assistant_prompt = f.read()

    st.session_state.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": assistant_prompt}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type Here!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["groq_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        def stream_text():
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        response = st.write_stream(stream_text())

    st.session_state.messages.append({"role": "assistant", "content": response})