import streamlit as st
import pandas as pd
import numpy as np
import os
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.backends.backend_agg import RendererAgg

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import FunctionTransformer

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer 
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA

#Loading the data
@st.cache
def get_data_regression():
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'house_price_prediction.csv'))

    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].replace(0, np.nan)
    df = df.replace({pd.NA: np.nan})
    df = df.drop(columns=['Id'])
    return df



def get_imputer(imputer):
    if imputer == 'None':
        return 'drop'
    if imputer == 'Most frequent value':
        return SimpleImputer(strategy='most_frequent', missing_values=np.nan)
    if imputer == 'Mean':
        return SimpleImputer(strategy='mean', missing_values=np.nan)
    if imputer == 'Median':
        return SimpleImputer(strategy='median', missing_values=np.nan)

def get_pipeline_missing_num(imputer, scaler):
    if imputer == 'None':
        return 'drop'
    if imputer == 'Mean':
        pipeline = make_pipeline(SimpleImputer(strategy='mean', missing_values=np.nan))
    if imputer == 'Median':
        pipeline = make_pipeline(SimpleImputer(strategy='median', missing_values=np.nan))
    if(scaler != 'None'):
        pipeline.steps.append(('scaling', get_scaling(scaler)))
    return pipeline


def get_pipeline_missing_cat(imputer, encoder):
    if imputer == 'None' or encoder == 'None':
        return 'drop'
    if imputer == 'Most frequent value':
        pipeline = make_pipeline(SimpleImputer(strategy='most_frequent', missing_values=np.nan))
    pipeline.steps.append(('encoding', get_encoding(encoder)))
    return pipeline

def get_encoding(encoder):
    if encoder == 'None':
        return 'drop'
    if encoder == 'Ordinal encoder':
        return OrdinalEncoder(handle_unknown='use_encoded_value')
    if encoder == 'OneHotEncoder':
        return OneHotEncoder(handle_unknown='ignore')
    if encoder == 'CountVectorizer':
        return CountVectorizer()

def get_scaling(scaler):
    if scaler == 'None':
        return 'passthrough'
    if scaler == 'Standard scaler':
        return StandardScaler()
    if scaler == 'MinMax scaler':
        return MinMaxScaler()
    if scaler == 'Robust scaler':
        return RobustScaler()

def get_ml_algorithm(algorithm, hyperparameters):
    if algorithm == 'Linear regression':
        return LinearRegression()
    
def get_dim_reduc_algo(algorithm, hyperparameters):
    if algorithm == 'None':
        return 'passthrough'
    if algorithm == 'PCA':
        return PCA(n_components = hyperparameters['n_components'])
    if algorithm == 'LDA':
        return LDA(solver = hyperparameters['solver'])
    if algorithm == 'Kernel PCA':
        return KernelPCA(n_components = hyperparameters['n_components'], kernel = hyperparameters['kernel'])
    

#def app():
    #configuration of the page
st.set_page_config(layout="wide")
    # matplotlib.use("agg")
    # _lock = RendererAgg.lock

SPACER = .2
ROW = 1

title_spacer1, title, title_spacer_2 = st.beta_columns((.1,ROW,.1))
with title:
    st.title('Regression exploratory tool')
    st.markdown("""
            This app allows you to test different machine learning algorithms and combinations of hyperparameters 
            to approximate the value of a house!
            The dataset is composed of house description and information and their price.
            * Use the menu on the left to select ML algorithm and hyperparameters
            * Data source (accessed mid may 2021): [house price dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=train.csv).
            * The code can be accessed at [code](https://github.com/max-lutz/ML-exploration-tool).
            """)
    st.write('')


df = get_data_regression()
#df.info()
#st.write(df)
target_selected = 'SalePrice'

#Sidebar 
#selection box for the different features
st.sidebar.header('Preprocessing')
missing_value_threshold_selected = st.sidebar.slider('Max missing values in feature (%)', 0,100,30,1)
categorical_imputer_selected = st.sidebar.selectbox('Handling categorical missing values', ['None', 'Most frequent value', 'Delete row'])
numerical_imputer_selected = st.sidebar.selectbox('Handling numerical missing values', ['None', 'Median', 'Mean', 'Delete row'])

encoder_selected = st.sidebar.selectbox('Encoding categorical values', ['None', 'OneHotEncoder'])
scaler_selected = st.sidebar.selectbox('Scaling', ['None', 'Standard scaler', 'MinMax scaler', 'Robust scaler'])
text_encoder_selected = st.sidebar.selectbox('Encoding text values', ['None', 'CountVectorizer'])

X = df.drop(columns = target_selected)
Y = df[target_selected].values.ravel()

row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.beta_columns((SPACER,ROW,SPACER,ROW, SPACER))

with row1_1:
    st.write(df)

with row1_2:
    number_features = len(X.columns)

    #feature with missing values
    drop_cols = []
    for col in X.columns:
        #put the feature in the drop trable if threshold not respected
        if(X[col].isna().sum()/len(X)*100 > missing_value_threshold_selected):
            drop_cols.append(col)
    

    #numerical columns
    num_cols_extracted = [col for col in X.select_dtypes(include='number').columns]
    num_cols = []
    num_cols_missing = []
    cat_cols = []
    cat_cols_missing = []
    for col in num_cols_extracted:
        if(len(X[col].unique()) < 25):
            cat_cols.append(col)
        else:
            num_cols.append(col)
           
    #categorical columns
    obj_cols = [col for col in X.select_dtypes(include=['object']).columns]
    text_cols = []
    text_cols_missing = []
    for col in obj_cols:
        if(len(X[col].unique()) < 25):
            cat_cols.append(col)
        else:
            text_cols.append(col)

    #text columns

    #remove dropped columns
    for element in drop_cols:
        if element in num_cols:
            num_cols.remove(element)
        if element in cat_cols:
            cat_cols.remove(element)
        if element in text_cols:
            text_cols.remove(element)

    #check if we miss any column
    # all_cols = [drop_cols, num_cols, cat_cols, text_cols]
    # all_cols_set = set()
    # for list_ in all_cols:
    #     for col in list_:
    #         if(col in all_cols_set):
    #             print('Warning, column ',col,' is duplicate')
    #         all_cols_set.add(col)
    # original_cols_set = set(X.columns)
    # badly_written_cols = all_cols_set - original_cols_set
    # missing_cols = original_cols_set - all_cols_set
    # print('Columns badly written :', badly_written_cols)
    # print('Missing columns :', missing_cols)

    #display info on dataset
    st.write('Original size of the dataset', X.shape)
    st.write('Dropping ', round(100*len(drop_cols)/number_features,2), '% of feature for missing values')
    st.write('Numerical columns : ', round(100*len(num_cols)/number_features,2), '%')
    st.write('Categorical columns : ', round(100*len(cat_cols)/number_features,2), '%')
    st.write('Text columns : ', round(100*len(text_cols)/number_features,2), '%')

    st.write('Total : ', round(100*(len(drop_cols)+len(num_cols)+len(cat_cols)+len(text_cols))/number_features,2), '%')
    
    #create new lists for columns with missing elements
    for col in X.columns:
        if (col in num_cols and X[col].isna().sum() > 0):
            num_cols.remove(col)
            num_cols_missing.append(col)
        if (col in cat_cols and X[col].isna().sum() > 0):
            cat_cols.remove(col)
            cat_cols_missing.append(col)
        if (col in text_cols and X[col].isna().sum() > 0):
            text_cols.remove(col)
            text_cols_missing.append(col)


#need to make two preprocessing pipeline too handle the case encoding without imputer...
preprocessing = make_column_transformer(
    (get_pipeline_missing_cat(categorical_imputer_selected, encoder_selected) , cat_cols_missing),
    (get_pipeline_missing_num(numerical_imputer_selected, scaler_selected) , num_cols_missing),

    (get_encoding(encoder_selected), cat_cols),
    (get_encoding(text_encoder_selected), text_cols),
    (get_scaling(scaler_selected), num_cols)
)

st.text(preprocessing)

st.sidebar.header('K fold cross validation selection')
nb_splits = st.sidebar.slider('Number of splits', min_value=3, max_value=20)
rdm_state = st.sidebar.slider('Random state', min_value=0, max_value=42)

st.sidebar.header('Model selection')
classifier_list = ['Linear regression']
classifier_selected = st.sidebar.selectbox('', classifier_list)

st.sidebar.header('Hyperparameters selection')
hyperparameters = {}

folds = KFold(n_splits=nb_splits, shuffle=True, random_state=rdm_state)

pipeline = Pipeline([
    ('preprocessing' , preprocessing),
    #('dimension reduction', get_dim_reduc_algo(dimension_reduction_alogrithm_selected, hyperparameters_dim_reduc)),
    ('ml', get_ml_algorithm(classifier_selected, hyperparameters))
])

cv_score = cross_val_score(pipeline, X, Y, cv=folds)
preprocessing.fit(X)
X_preprocessed = preprocessing.transform(X)
#pd.DataFrame(X_preprocessed).info()

with st.beta_expander("Dataframe preprocessed"):
    st.write(X_preprocessed)
    st.text(X_preprocessed)

st.subheader('Results')
st.write('Accuracy : ', round(cv_score.mean()*100,2), '%')
st.write('Standard deviation : ', round(cv_score.std()*100,2), '%')



# preprocessing = make_column_transformer(
#     (get_encoding(encoder_selected) , cat_cols),
#     (get_scaling(scaler_selected) , passthrough_cols)
# )
# dim = preprocessing.fit_transform(X).shape[1]
# if(encoder_selected == 'OneHotEncoder'):
#     dim = dim - 1

# st.sidebar.header('Dimension reduction')
# dimension_reduction_alogrithm_selected = st.sidebar.selectbox('Algorithm', ['None', 'PCA', 'LDA', 'Kernel PCA'])

# hyperparameters_dim_reduc = {}                                      
# if(dimension_reduction_alogrithm_selected == 'PCA'):
#     hyperparameters_dim_reduc['n_components'] = st.sidebar.slider('Number of components (default = nb of features - 1)', 2, dim, dim, 1)
# if(dimension_reduction_alogrithm_selected == 'LDA'):
#     hyperparameters_dim_reduc['solver'] = st.sidebar.selectbox('Solver (default = svd)', ['svd', 'lsqr', 'eigen'])
# if(dimension_reduction_alogrithm_selected == 'Kernel PCA'):
#     hyperparameters_dim_reduc['n_components'] = st.sidebar.slider('Number of components (default = nb of features - 1)', 2, dim, dim, 1)
#     hyperparameters_dim_reduc['kernel'] = st.sidebar.selectbox('Kernel (default = linear)', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
    

# st.sidebar.header('K fold cross validation selection')
# nb_splits = st.sidebar.slider('Number of splits', min_value=3, max_value=20)
# rdm_state = st.sidebar.slider('Random state', min_value=0, max_value=42)

# st.sidebar.header('Model selection')
# classifier_list = ['Logistic regression', 'Support vector', 'K nearest neighbors', 'Naive bayes', 'Ridge classifier', 'Decision tree', 'Random forest']
# classifier_selected = st.sidebar.selectbox('', classifier_list)

# st.sidebar.header('Hyperparameters selection')
# hyperparameters = {}

# if(classifier_selected == 'Logistic regression'):
#     hyperparameters['solver'] = st.sidebar.selectbox('Solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
#     if (hyperparameters['solver'] == 'liblinear'):
#         hyperparameters['penalty'] = st.sidebar.selectbox('Penalty (default = l2)', ['none', 'l1', 'l2'])
#     if (hyperparameters['solver'] == 'saga'):
#         hyperparameters['penalty'] = st.sidebar.selectbox('Penalty (default = l2)', ['none', 'l1', 'l2', 'elasticnet'])
#     else:
#         hyperparameters['penalty'] = st.sidebar.selectbox('Penalty (default = l2)', ['none', 'l2'])
#     hyperparameters['C'] = st.sidebar.selectbox('C (default = 1.0)', [100, 10, 1, 0.1, 0.01])

# if(classifier_selected == 'Ridge classifier'):
#     hyperparameters['alpha'] = st.sidebar.slider('Alpha (default value = 1.0)', 0.0, 10.0, 1.0, 0.1)
#     hyperparameters['solver'] = st.sidebar.selectbox('Solver (default = auto)', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
    
# if(classifier_selected == 'K nearest neighbors'):
#     hyperparameters['n_neighbors'] = st.sidebar.slider('Number of neighbors (default value = 5)', 1, 21, 5, 1)
#     hyperparameters['metric'] = st.sidebar.selectbox('Metric (default = minkowski)', ['minkowski', 'euclidean', 'manhattan', 'chebyshev'])
#     hyperparameters['weights'] = st.sidebar.selectbox('Weights (default = uniform)', ['uniform', 'distance'])

# if(classifier_selected == 'Support vector'):
#     hyperparameters['kernel'] = st.sidebar.selectbox('Kernel (default = rbf)', ['rbf', 'linear', 'poly', 'sigmoid'])
#     hyperparameters['C'] = st.sidebar.selectbox('C (default = 1.0)', [100, 10, 1, 0.1, 0.01])

# if(classifier_selected == 'Decision tree'):
#     hyperparameters['criterion'] = st.sidebar.selectbox('Criterion (default = gini)', ['gini', 'entropy'])
#     hyperparameters['min_samples_split'] = st.sidebar.slider('Min sample splits (default = 2)', 2, 20, 2, 1)

# if(classifier_selected == 'Random forest'):
#     hyperparameters['n_estimators'] = st.sidebar.slider('Number of estimators (default = 100)', 10, 500, 100, 10)
#     hyperparameters['criterion'] = st.sidebar.selectbox('Criterion (default = gini)', ['gini', 'entropy'])
#     hyperparameters['min_samples_split'] = st.sidebar.slider('Min sample splits (default = 2)', 2, 20, 2, 1)

# with st.beta_expander("Original dataframe"):
#     st.write(df)

# preprocessing = make_column_transformer(
#     (get_encoding(encoder_selected) , cat_cols),
#     (get_scaling(scaler_selected) , passthrough_cols)
# )

# folds = KFold(n_splits=nb_splits, shuffle=True, random_state=rdm_state)

# pipeline = Pipeline([
#     ('preprocessing' , preprocessing),
#     ('dimension reduction', get_dim_reduc_algo(dimension_reduction_alogrithm_selected, hyperparameters_dim_reduc)),
#     ('ml', get_ml_algorithm(classifier_selected, hyperparameters))
# ])

# cv_score = cross_val_score(pipeline, X, Y, cv=folds)
# preprocessing.fit(X)
# X_preprocessed = preprocessing.transform(X)


# with st.beta_expander("Dataframe preprocessed"):
#     st.write(X_preprocessed)


# st.subheader('Results')
# st.write('Accuracy : ', round(cv_score.mean()*100,2), '%')
# st.write('Standard deviation : ', round(cv_score.std()*100,2), '%')


