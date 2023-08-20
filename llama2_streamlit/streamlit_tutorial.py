import streamlit as st
import pandas as pd
from time import sleep
from datetime import time

st.title("Hi! I am Streamlit web app")
st.header("Hi, i am your header")
st.subheader("Hi, i am your subheader")
st.text("Hi, i am a text function and programmers use")
st.markdown("## Hello World\n `i am markdown`")
st.markdown("----")
st.markdown("[google](https://www.google.com)")
st.caption("Hi i am a caption")

json = {"a": "1,2,3", "b":"3,4,5"}
st.json(json)

code = """
def function():
    print("Hello World, this is streamlit function)
    return 0
"""
st.code(code, language="python")
st.write("## Write Function here")
st.metric(label="Wind Speed", value="129ms¹", delta="1.4ms¹")

table = pd.DataFrame({"Column1":[1,2,3,4,5,6,7], "Column2":[8,9,10,11,12,13,14]})
st.table(table)
st.dataframe(table)

st.image("techmon.jpg",caption="This is my drone project", width=680)
st.audio("05Officiallymissingyou.mp3")
#st.video("video.mp4")

# To remove streamlit footer
st.markdown("""
<style>
.css-cio0dv.ea3mdgi1
{
    visibility:hidden;
} 
</style>             
""", unsafe_allow_html=True)

# streamlit write the information
state = st.checkbox("Checkbox", value=True)
if state:
    st.write("Hi")
else:
    pass

# Print the Changed v1
def change_v1():
    print("Changed v1")
state = st.checkbox("Checkbox_1", value=True, on_change=change_v1)

# True of False
def change_v2():
    print(st.session_state.checker)
state = st.checkbox("Checkbox_2", value=True, on_change=change_v2, key="checker")

radio_btn = st.radio("In which country do you live ?", options=("US", "Sarawak", "Sabah"))

def btn_click():
    print("Button clicked")
    
btn = st.button("Click Me !", on_click=btn_click())

select = st.selectbox("What is your favourite car ?", options=("Saga","Axia","X50"),index=0)
multi_select = st.multiselect("What is your favourite car ?", options=("Saga","Axia","X50"))
st.write(multi_select)

st.title("Uploading Files")
st.markdown("---")

image = st.file_uploader("Please upload an image", type=["png","jpg"])
if image is not None:
    st.image(image)
    

images = st.file_uploader("Please upload an image", type=["png","jpg"], accept_multiple_files=True)
if images is not None:
    for image in images:
        st.image(image)
        
val = st.slider("This is a Slider", min_value=50, max_value=150, value=70)
print(val)

user_input = st.text_input("Enter your Course Title here")
st.write(user_input)

course_descr = st.text_area("Course Description")
st.write(course_descr)

user_date = st.date_input("Enter your registraion Date")
user_time = st.time_input("Set Timer")

# bar = st.progress(0)
# for i in range(10):
#     bar.progress((i*1)*10)
#     sleep(1)
    
new_val = st.time_input("Set Timer",value=time(0,0,0))
print(new_val)

def converter(value):
    m,s,mm = value.split(":")
    t_s = int(m)*60 + int(s)+int(mm)/1000
    return t_s

if str(new_val) == "00:00:00":
    st.write("Please set timer")
else:
    sec = converter(str(new_val))
    bar = st.progress(0)
    per = sec/100
    progress_status = st.empty()
    for i in range(100):
        bar.progress(i+1)
        progress_status.write(str(i) + " %")
        sleep(per)
    
