### hello_streamlit.py Introduces different features of Streamlit

# Import needed libraries
import streamlit as st
import datetime as dt

# Set title
st.title('Hello, Streamlit!')

# Get the name
name = st.text_input("First Name",value="Coding Dojo")

# Add Date Selection 
date = st.date_input(label="Date of Birth", min_value=dt.date(1900,1,1),value=dt.date(2013,1,1))

# Create the select box with all options
season = st.selectbox("Favorite Season", ['Winter','Spring','Summer', 'Fall'])

# Adding user excitment
excitement = st.slider("How excited are you to learn Streamlit?? (1=Not at all; 10=Very!)", min_value=1, max_value=10,value=5)

# If the button is pressed, print message
if st.button("Introduce me!"):
    st.markdown(f"""
    > #### ***Hello, World!*** My name is **{name}**.
    - I was born on **{date}**.
    - My favorite season is **{season}**.
    - My excitement for learning streamlit is... **{excitement}/10.**""")