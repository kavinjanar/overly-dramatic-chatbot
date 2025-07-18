from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage

import streamlit as st
from streamlit_chat import message
import os
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

system_message = """You are Drama Bot, a chatbot that answers user questions in a dramatic and engaging way. Try to dramatize your responses and make them entertaining
 however, do not make your resspones too long."""



gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=gemini_model)

conversation.memory.chat_memory.add_message(
    SystemMessage(content=system_message)
)

st.title("Overly-Dramatic Chatbot")
st.subheader("It's a chatbot. Just way too overly dramatic :)")

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversation.predict(input = prompt)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message) # Add response to message history