### ames_EDA.py for Wk3. Dataframes and String Outputs

# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO


# Add title
st.title("EDA with Streamlit")
# Define the columns you want to use 
columns_to_use = ['SalePrice', 'Living Area Sqft', 'Lot Frontage', 'Bldg Type', 'Bedroom',
                    'Total Full Baths','MS Zoning','Street', 
                    'Alley','Utilities']

# Function for loading data
# Adding data caching
@st.cache_data
def load_data():
    fpath =  "Data/ames-housing-dojo-for-ml.csv"
    df = pd.read_csv(fpath)
    df = df.set_index("PID")
    df = df[columns_to_use]
    return df

# load the data 
df = load_data()

##################################

# Display an interactive dataframe
st.header("Displaying a DataFrame")
st.dataframe(df, width=800)

# Display descriptive statistics
st.markdown('#### Descriptive Statistics')
st.dataframe(df.describe().round(2))

# Capture .info()
# Create a string buffer to capture the content
buffer = StringIO()
# Write the info into the buffer
df.info(buf=buffer)
# Retrieve the content from the buffer
summary_info = buffer.getvalue()
# Use Streamlit to display the info
st.markdown("#### Summary Info")
st.text(summary_info)

# We could display the output series as a dataframe
st.markdown("#### Null Values as dataframe")
nulls =df.isna().sum()
st.dataframe(nulls)
# Create a string buffer to capture the content
buffer = StringIO()
# Write the content into the buffer...use to_string
df.isna().sum().to_string(buffer)
# Retrieve the content from the buffer
null_values = buffer.getvalue()
# Use Streamlit to display the info
st.markdown("#### Null Values as String")
st.text(null_values)