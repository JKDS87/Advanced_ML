### Streamlit_Project_1_Core_B.py for Wk3 of Advanced Machine Learning
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
## Part A - Basic buttons and info about the dataset
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


##################################
## Part B - Explore selectable features with appropriate plots or figures
##################################

## explore_categorical/explore_numeric Example

# Header
st.markdown("#### Displaying an appropriate plot based on selected column:")

# Functions
def explore_categorical(df, x, fillna = True, placeholder = 'MISSING',
                        figsize = (6,4), order = None):
  # Make a copy of the dataframe and fillna 
  temp_df = df.copy()
  # Before filling nulls, save null value counts and percent for printing 
  null_count = temp_df[x].isna().sum()
  null_perc = null_count/len(temp_df)* 100
  # fillna with placeholder
  if fillna == True:
    temp_df[x] = temp_df[x].fillna(placeholder)
  # Create figure with desired figsize
  fig, ax = plt.subplots(figsize=figsize)
  # Plotting a count plot 
  sns.countplot(data=temp_df, x=x, ax=ax, order=order)
  # Rotate Tick Labels for long names
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
  # Add a title with the feature name included
  ax.set_title(f"Column: {x}")
  
  # Fix layout and show plot (before print statements)
  fig.tight_layout()
  plt.show()
  return fig, ax

def explore_numeric(df, x, figsize=(6,5) ):
  """Source: https://login.codingdojo.com/m/606/13765/117605"""
  # Making our figure with gridspec for subplots
  gridspec = {'height_ratios':[0.7,0.3]}
  fig, axes = plt.subplots(nrows=2, figsize=figsize,
                           sharex=True, gridspec_kw=gridspec)
  # Histogram on Top
  sns.histplot(data=df, x=x, ax=axes[0])
  # Boxplot on Bottom
  sns.boxplot(data=df, x=x, ax=axes[1])
  ## Adding a title
  axes[0].set_title(f"Column: {x}", fontweight='bold')
  ## Adjusting subplots to best fill Figure
  fig.tight_layout()
  # Ensure plot is shown before message
  plt.show()
  return fig

## Show a selectbox to pick a feature, then display 
## the appropriate figure
# Add a selectbox for all possible features
column = st.selectbox(label="Select a column", options=df.columns.tolist())
# Conditional statement to determine which function to use
if df[column].dtype == 'object':
    fig, ax  = explore_categorical(df, column)
else:
    fig = explore_numeric(df, column)
    
# Display appropriate eda plots
st.pyplot(fig)

##################################

## plot_numeric_vs_target/plot_categorical_vs_target Example

# Header
st.markdown("#### Plotting Item Features vs Outlet Sales - Seaborn")

# Functions
def plot_categorical_vs_target(df, x, y='Item_Outlet_Sales',figsize=(6,4),
                            fillna = True, placeholder = 'MISSING',
                            order = None):
  # Make a copy of the dataframe and fillna 
  temp_df = df.copy()
  # fillna with placeholder
  if fillna == True:
    temp_df[x] = temp_df[x].fillna(placeholder)
  # or drop nulls prevent unwanted 'nan' group in stripplot
  else:
    temp_df = temp_df.dropna(subset=[x]) 
  # Create the figure and subplots
  fig, ax = plt.subplots(figsize=figsize)
  
  # Barplot 
  sns.barplot(data=temp_df, x=x, y=y, ax=ax, order=order, alpha=0.6,
              linewidth=1, edgecolor='black', errorbar=None)
  # Boxplot
  sns.stripplot(data=temp_df, x=x, y=y, hue=x, ax=ax, 
                order=order, hue_order=order, legend=False,
                edgecolor='white', linewidth=0.5,
                size=3,zorder=0)
  # Rotate xlabels
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
  # Add a title
  ax.set_title(f"{x} vs. {y}")
  fig.tight_layout()
  return fig, ax

def plot_numeric_vs_target(df, x, y='Item_Outlet_Sales', figsize=(6,4), **kwargs): # kwargs for sns.regplot
  # Calculate the correlation
  corr = df[[x,y]].corr().round(2)
  r = corr.loc[x,y]
  # Plot the data
  fig, ax = plt.subplots(figsize=figsize)
  scatter_kws={'ec':'white','lw':1,'alpha':0.8}
  sns.regplot(data=df, x=x, y=y, ax=ax, scatter_kws=scatter_kws, **kwargs) # Included the new argument within the sns.regplot function
  ## Add the title with the correlation
  ax.set_title(f"{x} vs. {y} (r = {r})")
  # Make sure the plot is shown before the print statement
  plt.show()
  return fig, ax

## Add a selectbox for all possible features
cols = df.columns.tolist()
features_to_use = cols[:]
target = 'Item_Outlet_Sales'
features_to_use.remove(target)
feature = st.selectbox(label="Select a feature", options=df.columns.tolist())

# Conditional statement to determine which function to use
if df[feature].dtype == 'object':
    fig, ax  = plot_categorical_vs_target(df, x=feature)
else:
    fig, ax = plot_numeric_vs_target(df, x=feature)
    
# Display appropriate eda plots
st.pyplot(fig)