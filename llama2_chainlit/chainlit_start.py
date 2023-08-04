import chainlit as cl

# Continously on loop and read the message from user input
@cl.on_message
async def main(message: str):
    result = message

    await cl.Message(content=f"Sure, here is the message {result}").send()