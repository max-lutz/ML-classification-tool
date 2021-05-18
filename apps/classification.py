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
st.sidebar.header('Preprocessing')

classifier_selected = st.sidebar.selectbox('Scaling', ['None', 'Standard scaler', 'MinMax scaler', 'Robust scaler'], 
                                            help='Scaling data can improve the performance of ML algorithms.')


st.sidebar.header('Model selection')
classifier_list = ['Logistic regression', 'Support vector', 'Kernel support', 'K neirest neighbors', 'Naive bayes', 'Decision tree', 'Random forest']
classifier_selected = st.sidebar.selectbox('Select regression algorithm', classifier_list)

title_spacer1, title, title_spacer_2 = st.beta_columns((.1,ROW,.1))
with title:
    st.title('Classification exploratory tool')
    st.markdown("""
            This app allows you to test different machine learning algorithms and combinations of hyperparameters 
            to classify patients with risk of developping heart diseases!
            The dataset is composed of medical observation of patients and their risk of developping heart diseases
            * Use the menu on the left to select ML algorithm and hyperparameters
            * Data source (accessed mid may 2021): [heart disease dataset](https://ieee-dataport.org/open-access/heart-disease-dataset-comprehensive).
            * The code can be accessed at [code](https://github.com/max-lutz/ML-exploration-tool).
            """)

st.write(df_classification)

st.write(df_classification.corr())