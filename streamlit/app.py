import streamlit as st
import ollama    # to access ollama models

st.title("Demo Chatbot")

# CHOOSE MODEL
model_option = st.selectbox(label="Choose a model: ", options=["llama2-uncensored:7b", "llama3.2:3b"])

# EMPTY STATE CHECK
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi, How can I help you?"}]

# PRINT SESSION HISTORY
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])   


# STREAM TOKENS
def generate_response():
    response = ollama.chat(
        # model="llama2-uncensored:7b",
        model=model_option,
        messages=st.session_state.messages,
        stream=True 
    )

    for chunk in response:
        token = chunk["message"]["content"]
        st.session_state["full_message"] += token    # intermediate variable
        yield token


# TAKE USER INPUT AND RESPOND
if prompt:= st.chat_input("Write Something"):

    # add current user prompt to session state
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # write user message to screen
    st.chat_message("user").write(prompt)

    # create intermediate full message variable
    st.session_state["full_message"] = ""

    # write assistant response to screen
    st.chat_message("assistant").write_stream(generate_response)

    # add assistant response to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": st.session_state["full_message"]
    })