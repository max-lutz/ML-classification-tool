import streamlit as st
import pandas as pd
import numpy as np
import os
from matplotlib.backends.backend_agg import RendererAgg

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer 
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
  


#Loading the data
@st.cache
def get_data_classification():
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'heart_statlog.csv'))
    df.loc[df['chest pain type'] == 1, 'chest pain type'] = 'typical angina'
    df.loc[df['chest pain type'] == 2, 'chest pain type'] = 'atypical angina'
    df.loc[df['chest pain type'] == 3, 'chest pain type'] = 'non-anginal pain'
    df.loc[df['chest pain type'] == 4, 'chest pain type'] = 'asymptomatic'
    df['chest pain type'] = df['chest pain type'].astype(str)

    df.loc[df['sex'] == 1, 'sex'] = 'male'
    df.loc[df['sex'] == 0, 'sex'] = 'female'
    df['sex'] = df['sex'].astype(str)

    df.loc[df['resting ecg'] == 0, 'resting ecg'] = 'normal'
    df.loc[df['resting ecg'] == 1, 'resting ecg'] = 'ST-T wave abnormality'
    df.loc[df['resting ecg'] == 2, 'resting ecg'] = 'probable or definite left ventricular hypertrophy'
    df['resting ecg'] = df['resting ecg'].astype(str)

    df.loc[df['exercise angina'] == 0, 'exercise angina'] = 'no'
    df.loc[df['exercise angina'] == 1, 'exercise angina'] = 'yes'
    df['exercise angina'] = df['exercise angina'].astype(str)

    df.loc[df['ST slope'] == 0, 'ST slope'] = 'unsloping'
    df.loc[df['ST slope'] == 1, 'ST slope'] = 'flat'
    df.loc[df['ST slope'] == 2, 'ST slope'] = 'downslopping'
    df['ST slope'] = df['ST slope'].astype(str)

    return df

def get_encoding(encoder):
    if encoder == 'None':
        return 'drop'
    if encoder == 'Ordinal encoder':
        return OrdinalEncoder(handle_unknown='use_encoded_value')
    if encoder == 'OneHotEncoder':
        return OneHotEncoder(handle_unknown='ignore')

def get_scaling(scaler):
    if scaler == 'None':
        return 'passthrough'
    if scaler == 'Standard scaler':
        return StandardScaler()
    if scaler == 'MinMax scaler':
        return MinMaxScaler()
    if scaler == 'Robust scaler':
        return RobustScaler()

def get_ml_algorithm(algorithm, hyperparameters=None):
    if algorithm == 'Logistic regression':
        return LogisticRegression()
    if algorithm == 'Support vector':
        return SVC()
    if algorithm == 'Naive bayes':
        return GaussianNB()
    if algorithm == 'K nearest neighbors':
        return KNeighborsClassifier()
    if algorithm == 'Ridge classifier':
        return RidgeClassifier()
    if algorithm == 'Decision tree':
        return DecisionTreeClassifier()
    if algorithm == 'Random forest':
        return RandomForestClassifier()

#def app():
    #configuration of the page
st.set_page_config(layout="wide")

SPACER = .2
ROW = 1

df_classification = get_data_classification()
X = df_classification.drop(columns = 'target')
Y = df_classification['target'].values.ravel()

passthrough_cols = ['age', 'resting bp s', 'cholesterol', 'fasting blood sugar', 'max heart rate', 'oldpeak']
cat_cols = ['resting ecg', 'exercise angina', 'ST slope', 'chest pain type', 'sex']


# Sidebar 
#selection box for the different features
st.sidebar.header('Preprocessing')
encoder_selected = st.sidebar.selectbox('Encoding', ['None', 'OneHotEncoder'], 
                                            help='Encoding allow ML algorithms to understand categorical features.')

scaler_selected = st.sidebar.selectbox('Scaling', ['None', 'Standard scaler', 'MinMax scaler', 'Robust scaler'], 
                                            help='Scaling data can improve the performance of ML algorithms.')

st.sidebar.header('K fold cross validation selection')
nb_splits = st.sidebar.slider('Number of splits', min_value=3, max_value=20)
rdm_state = st.sidebar.slider('Random state', min_value=0, max_value=42)

st.sidebar.header('Model selection')
classifier_list = ['Logistic regression', 'Support vector', 'K nearest neighbors', 'Naive bayes', 'Ridge classifier', 'Decision tree', 'Random forest']
classifier_selected = st.sidebar.selectbox('', classifier_list)

st.sidebar.header('Hyperparameters selection')
hyperparameters = {}


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


#get_encoding(encoder_selected)

preprocessing = make_column_transformer(
    (get_encoding(encoder_selected) , cat_cols),
    (get_scaling(scaler_selected) , passthrough_cols)
)

folds = KFold(n_splits=nb_splits, shuffle=True, random_state=rdm_state)

pipeline = Pipeline([
    ('preprocessing' , preprocessing),
    ('ml', get_ml_algorithm(classifier_selected))
])

cv_score = cross_val_score(pipeline, X, Y, cv=folds)

st.write(X)


preprocessing.fit(X)
X_preprocessed = preprocessing.transform(X)

st.write(X_preprocessed)

st.write(cv_score.mean(), cv_score.std())


#st.write(df_classification.corr())