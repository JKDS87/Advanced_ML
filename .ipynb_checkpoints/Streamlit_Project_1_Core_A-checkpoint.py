### Streamlit_Project_1_Core_A.py for Wk3 of Advanced Machine Learning
# Create a simple app to demonstrate the use of Streamlit features

# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Function for loading data
# Adding data caching
@st.cache_data
def load_data():
    fpath =  "Data/sales_2023_cleaned.csv"
    df = pd.read_csv(fpath)
    return df

# load the data 
df = load_data()

##################################

# Add title
st.title("Sales Price Analysis")

# Display an interactive dataframe
st.header("Product Sales Data")
st.dataframe(df, width=800)

# Display Descriptive Statistics button
st.markdown('#### Descriptive Statistics')
if st.button('Show Descriptive Statistics'):
    st.dataframe(df.describe().round(2))

## Display Summary Information button
# Create a string buffer to capture content and write the info into the buffer
buffer = StringIO()
df.info(buf=buffer)
summary_info = buffer.getvalue()
st.markdown("#### Summary Info")
if st.button('Show Summary Info'):
    st.text(summary_info)

## Display Null Values button
st.markdown("#### Null Values")
if st.button('Show Null Values'):
    nulls =df.isna().sum()
    st.dataframe(nulls)
