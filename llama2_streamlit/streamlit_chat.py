import streamlit as st
import numpy as np

# Chat Message
# Call with 'with'
with st.chat_message("user"):
    st.write("Hello 👋")
    st.line_chart(np.random.randn(30, 3))

# Call directly using object
message = st.chat_message("assistant")
message.write("Hello human")
message.bar_chart(np.random.randn(30, 3))

# Chat Input
prompt = st.chat_input("Say Something")
if prompt:
    st.write(f"The user has sent: {prompt}")


