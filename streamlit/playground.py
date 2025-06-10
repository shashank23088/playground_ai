import numpy as np
import random
import time
# from openai import OpenAI    # fck openai
import streamlit as st

# with st.chat_message("user"):
#     st.write("Hello Bot!")

# message = st.chat_message("assistant")
# message.write("Hello Human!")
# message.bar_chart(np.random.randn(30, 3))

# prompt = st.chat_input("Say Something")
# if prompt:
#     response = st.chat_message("user")
#     response.write(prompt)

# session state stores chat history (state persistance and manipulation) [https://docs.streamlit.io/develop/concepts/architecture/session-state]


# streamed respnse emulator
def response_generator():
    repsonse = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi Human! Is there anything I can help you with?",
            "Do you need help?"
        ]
    )

    for word in response.split(" "):
        yield word + " "    # yield used to return a value from a function and yet continue to execute furthur lines
        time.sleep(0.10)    # streaming effect


st.title("Chat Bot")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# display chat messages on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# react to user input
# the walrus operator can also be used -> if prompt := st.chat_input("Say Something"):
prompt = st.chat_input("Speak UP!")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    # add user message to chat_history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # response = f"Echo: {prompt}"
    # with st.chat_message("assistant"):
    #     response = st.write_stream(response_generator())
    #     # st.markdown(response)
        
    # # add assistant message to chat_history
    # st.session_state.messages.append({
    #     "role": "assistant", 
    #     "content": response
    # })

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True
        )

        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
    

