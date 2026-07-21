import streamlit as st
from groq import Groq
from pathlib import Path
from types import SimpleNamespace
from pages.tools import get_tools_schema, execute_tool_call

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

@st.cache_resource
def load_tools():
    return get_tools_schema()

tools_schema = load_tools()
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
    if message["role"] in ("system", "tool"):
        continue
    if message["role"] == "assistant" and not message.get("content"):
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

MAX_TOOL_ROUNDS = 4

def call_model():
    return client.chat.completions.create(
        model=st.session_state["groq_model"],
        messages=[
            {k: v for k, v in m.items() if k in ("role", "content", "tool_call_id", "name", "tool_calls")}
            for m in st.session_state.messages
        ],
        stream=True,
        tools=[tools_schema]
    )

def consume_stream(stream, collected_tool_calls):
    """Yield content text as it streams in, and assemble tool-call deltas
    (which Groq sends split across multiple chunks) into collected_tool_calls."""
    for chunk in stream:
        delta = chunk.choices[0].delta

        if delta.content:
            yield delta.content

        for tool_call in (delta.tool_calls or []):
            entry = collected_tool_calls.setdefault(
                tool_call.index, {"id": None, "name": None, "arguments": ""}
            )
            if tool_call.id:
                entry["id"] = tool_call.id
            if tool_call.function and tool_call.function.name:
                entry["name"] = tool_call.function.name
            if tool_call.function and tool_call.function.arguments:
                entry["arguments"] += tool_call.function.arguments

# Accept user input
if prompt := st.chat_input("Type Here!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = ""
        for _ in range(MAX_TOOL_ROUNDS):
            tool_call_fragments = {}
            response = st.write_stream(consume_stream(call_model(), tool_call_fragments))

            if not tool_call_fragments:
                break

            st.session_state.messages.append({
                "role": "assistant",
                "content": response or "",
                "tool_calls": [
                    {
                        "id": frag["id"],
                        "type": "function",
                        "function": {"name": frag["name"], "arguments": frag["arguments"]},
                    }
                    for frag in tool_call_fragments.values()
                ],
            })

            for frag in tool_call_fragments.values():
                tool_call = SimpleNamespace(
                    id=frag["id"],
                    function=SimpleNamespace(name=frag["name"], arguments=frag["arguments"]),
                )
                function_response = execute_tool_call(tool_call)
                st.session_state.messages.append({
                    "role": "tool",
                    "tool_call_id": frag["id"],
                    "name": frag["name"],
                    "content": str(function_response)
                })

    st.session_state.messages.append({"role": "assistant", "content": response})