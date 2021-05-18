import streamlit as st
import pandas as pd
import base64
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.backends.backend_agg import RendererAgg
from datetime import date
  


#Loading the data
@st.cache
def get_data_classification():
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'heart_statlog.csv'))
    return df

#def app():
    #configuration of the page
st.set_page_config(layout="wide")
matplotlib.use("agg")
_lock = RendererAgg.lock

SPACER = .2
ROW = 1

df_classification = get_data_classification()

# Sidebar 
#selection box for the different features
st.sidebar.header('Select what to display')
classifier_list = ['Logistic regression', 'Support vector', 'Kernel support', 'K neirest neighbors', 'Naive bayes', 'Decision tree', 'Random forest']

title_spacer1, title, title_spacer_2 = st.beta_columns((.1,ROW,.1))
with title:
    st.title('Classification exploratory tool')

st.write(df_classification)

st.write(df_classification.corr())