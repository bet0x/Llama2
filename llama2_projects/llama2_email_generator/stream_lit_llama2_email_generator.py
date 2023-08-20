import streamlit as st
from streamlit_chat import message
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain import LLMChain

MODEL_PATH = r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q5_K_M.bin"

def getLLMResponse(form_input,email_sender,email_receipient,email_style):
    
    #llm = OpenAI(temperature=.9, model="text-davinci-003")

    # Wrapper for Llama-2-7B-Chat, Running Llama 2 on CPU

    #Quantization is reducing model precision by converting weights from 16-bit floats to 8-bit integers, 
    #enabling efficient deployment on resource-limited devices, reducing model size, and maintaining performance.

    #C Transformers offers support for various open-source models, 
    #among them popular ones like Llama, GPT4All-J, MPT, and Falcon.

    #C Transformers is the Python library that provides bindings for transformer models implemented in C/C++ using the GGML library
    llm = CTransformers(model=MODEL_PATH,
                    model_type='llama',
                    config={'max_new_tokens': 2048,
                            'temperature': 0.7,
                            'gpu_layers': 35,
                            'stream' : True,
                            },
                   )
    #Template for building the PROMPT
    template = """
    Write a email with {style} style and includes topic :{email_topic}.\n\nSender: {sender}\nRecipient: {recipient}
    \n\nEmail Text:
    
    """
    
    #Creating the final PROMPT
    prompt = PromptTemplate(input_variables=["style","email_topic","sender","recipient"], template=template)
    x = prompt.format(style=email_style, email_topic=form_input, sender=email_sender, recipient=email_receipient)
    
    #Generating the response using LLM
    #response = llm(prompt.format(style=email_style, email_topic=form_input, sender=email_sender, recipient=email_receipient))
    response = llm.predict(x)
    
    return response
    
st.set_page_config(page_title="Generate Emails",
                   page_icon='📧',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Email 📧")

form_input = st.text_area('Enter the email topic', height=275)

# Creating volumes for the UI -To receive input from user
col1, col2, col3 = st.columns([10,10,5])
with col1:
    email_sender = st.text_input("Sender Name")

with col2:
    email_receipient = st.text_input("Recipient Name")

with col3:
    email_style = st.selectbox("Writing Style", ("Formal", "Appreciating", "Not Satisfied", "Neutral", "Romantic"), index=0)
    print(email_style)
    
submit = st.button("Generate")


# When 'Generate' button is clicked 
if submit:
    st.write(getLLMResponse(form_input,email_sender,email_receipient,email_style))
   

    
    