import streamlit as st
from time import sleep

left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")


# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)

###################################

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted ?",
    ("Email", "Home Phone", "Mobile Phone")
)

with st.sidebar:
    selectbox = st.sidebar.selectbox(
        "How would you like to be contacted ?",
        ("Email", "Phone")
    )

    with st.echo():
        st.write("This code will be printed to the sidebar.")
    
    with st.spinner("Loading..."):
        sleep(5)
    st.success("Done !")
        

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5) days")
    )