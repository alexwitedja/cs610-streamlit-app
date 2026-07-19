import streamlit as st
from groq import Groq
from pathlib import Path

PROMPTS_DIR = Path().resolve() / "prompts"

st.title("Kiasu Care")

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
    st.session_state.messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "assistant",
            "content": """
                I'll ask you a few quick questions, then give you a rough sense of how urgent things look and what your
                options are. Takes about a minute.\n
                First — **what's bothering you today?** Describe it in your own words, like you'd tell a friend.
            """
        }
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