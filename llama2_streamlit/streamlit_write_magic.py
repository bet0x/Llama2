import streamlit as st
import pandas as pd

## Using Write
st.write('Hello, *World!* :sunglasses:')


st.write(1234)
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40],
}))


data_frame = ({"column1": [1,2,3,4,5], "column2":[1,2,3,4,5]})
st.write('1 + 1 = ', 2)
st.write('Below is a DataFrame:', data_frame, 'Above is a dataframe.')

## Using Magic
st.text("Using Magic")
df = pd.DataFrame({"column1":[1,2,3]})
df

x = 10
'x',x  # Draw the string 'x' and then the value of x

