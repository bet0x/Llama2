import chainlit as cl

# Initial message to start - This function will start first before others
@cl.on_chat_start
async def start():
    # 1st - This message will get display first
    msg = cl.Message(content="Starting the bot...")
    await msg.send()

    # 2nd - Then we update the content of the message
    msg.content = "Hi, Welcome to X-Fab Hotline Bot. What is your query ?"
    await msg.send()

# This function will be executed once `Start` function has completed
# Continously on loop and read the message from user input
@cl.on_message
async def main(message: str):
    result = message
    await cl.Message(content=f"Sure, here is the message {result}").send()
    
