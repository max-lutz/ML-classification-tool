import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA, TruncatedSVD
from sklearn.datasets import load_iris, load_diabetes, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import streamlit_download_button as button

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


@st.cache_data
def get_data_heart_disease():
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


def get_data_titanic():
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'titanic.csv'))
    target = df.pop("Survived")
    df.insert(len(df.columns), "Survived", target)
    return df


def get_imputer(imputer):
    if imputer == 'None':
        return 'drop'
    if imputer == 'Most frequent value':
        return SimpleImputer(strategy='most_frequent', missing_values=np.nan)
    if imputer == 'Constant value: "missing_value"':
        return SimpleImputer(strategy='constant', missing_values=np.nan, fill_value="missing_value")
    if imputer == 'Constant value: -1':
        return SimpleImputer(strategy='constant', missing_values=np.nan, fill_value=-1)
    if imputer == 'Mean':
        return SimpleImputer(strategy='mean', missing_values=np.nan)
    if imputer == 'Median':
        return SimpleImputer(strategy='median', missing_values=np.nan)


def get_pipeline_missing_num(imputer, scaler):
    if imputer == 'None':
        return 'drop'
    if imputer == 'Mean' or imputer == 'Median' or imputer == 'Most frequent value' or imputer == 'Constant value: -1':
        pipeline = make_pipeline(get_imputer(imputer))
    if (scaler != 'None'):
        pipeline.steps.append(('scaling', get_scaling(scaler)))
    return pipeline


def get_pipeline_missing_cat(imputer, encoder):
    if imputer == 'None' or encoder == 'None':
        return 'drop'
    if imputer == 'Most frequent value' or imputer == 'Constant value: "missing_value"':
        pipeline = make_pipeline(get_imputer(imputer))
    if (imputer != 'None'):
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
    if encoder == 'TfidfVectorizer':
        return TfidfVectorizer()


def get_scaling(scaler):
    if scaler == 'None':
        return 'passthrough'
    if scaler == 'Standard scaler':
        return StandardScaler()
    if scaler == 'MinMax scaler':
        return MinMaxScaler()
    if scaler == 'Robust scaler':
        return RobustScaler()


def convert_none(object):
    if (object == 'none' or object == 'None'):
        return None
    return object


def get_metric_name(metric, dict_required=False):
    metric_name = metric.lower()
    if (dict_required):
        return {metric_name: metric_name}
    else:
        return metric_name


def get_ml_algorithm(algorithm, hyperparameters):
    if algorithm == 'Logistic regression':
        return LogisticRegression(solver=hyperparameters['solver'], penalty=convert_none(hyperparameters['penalty']), C=hyperparameters['C'])
    if algorithm == 'Support vector':
        return SVC(kernel=hyperparameters['kernel'], C=hyperparameters['C'])
    if algorithm == 'Ridge':
        return RidgeClassifier(alpha=hyperparameters['alpha'], solver=hyperparameters['solver'])
    if algorithm == 'K nearest neighbors':
        return KNeighborsClassifier(n_neighbors=hyperparameters['n_neighbors'], metric=hyperparameters['metric'], weights=hyperparameters['weights'])
    if algorithm == 'Decision tree':
        return DecisionTreeClassifier(criterion=hyperparameters['criterion'], min_samples_split=hyperparameters['min_samples_split'])
    if algorithm == 'Random forest':
        return RandomForestClassifier(n_estimators=hyperparameters['n_estimators'], criterion=hyperparameters['criterion'], min_samples_split=hyperparameters['min_samples_split'])
    if algorithm == 'XGBoost':
        return XGBClassifier(n_estimators=hyperparameters['n_estimators'], max_depth=hyperparameters['max_depth'],
                             learning_rate=hyperparameters['learning_rate'], booster=hyperparameters['booster'])
    if algorithm == 'LightGBM':
        return LGBMClassifier(num_leaves=hyperparameters['num_leaves'], max_depth=hyperparameters['max_depth'],
                              learning_rate=hyperparameters['learning_rate'])


def get_dim_reduc_algo(algorithm, hyperparameters):
    if algorithm == 'None':
        return 'passthrough'
    if algorithm == 'PCA':
        return PCA(n_components=hyperparameters['n_components'])
    if algorithm == 'LDA':
        return LDA(solver=hyperparameters['solver'])
    if algorithm == 'Kernel PCA':
        return KernelPCA(n_components=hyperparameters['n_components'], kernel=hyperparameters['kernel'])
    if algorithm == 'Truncated SVD':
        return TruncatedSVD(n_components=hyperparameters['n_components'])


def get_fold(algorithm, nb_splits):
    if algorithm == 'KFold':
        return KFold(n_splits=nb_splits, shuffle=True, random_state=0)
    if algorithm == 'StratifiedKFold':
        return StratifiedKFold(n_splits=nb_splits, shuffle=True, random_state=0)


def split_columns(df, drop_cols=[]):
    # numerical columns
    num_cols_extracted = [col for col in df.select_dtypes(include='number').columns if col not in drop_cols]
    num_cols = []
    num_cols_missing = []
    cat_cols = []
    cat_cols_missing = []
    for col in num_cols_extracted:
        if (len(df[col].unique()) < 15):
            cat_cols.append(col)
        else:
            num_cols.append(col)

    # categorical columns
    obj_cols = [col for col in df.select_dtypes(exclude=['number']).columns if col not in drop_cols]
    text_cols = []
    for col in obj_cols:
        if (len(df[col].unique()) < 25):
            cat_cols.append(col)
        else:
            text_cols.append(col)

    return num_cols, cat_cols, text_cols, num_cols_missing, cat_cols_missing


def load_dataset(dataset):
    df = None
    if (dataset == 'Load my own dataset'):
        uploaded_file = st.file_uploader('File uploader')
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
    elif (dataset == 'Titanic dataset'):
        df = get_data_titanic()
    elif (dataset == 'Heart disease dataset'):
        df = get_data_heart_disease()
    elif (dataset == 'Iris dataset'):
        df = load_iris(as_frame=True).data
        df["target"] = load_iris(as_frame=True).target
    elif (dataset == 'Diabetes dataset'):
        df = load_diabetes(as_frame=True).data
        df["target"] = load_diabetes(as_frame=True).target
    elif (dataset == 'Wine dataset'):
        df = load_wine(as_frame=True).data
        df["target"] = load_wine(as_frame=True).target
    return df


def wrapper_selectbox(label, options, visible=True):
    if (not visible):
        return 'None'
    return st.sidebar.selectbox(label, options)


def calculate_test_score(metric_name, y_true, y_pred):
    if (metric_name == 'accuracy'):
        score = accuracy_score(y_true=y_true, y_pred=y_pred)
    elif (metric_name == 'f1'):
        score = np.sqrt(f1_score(y_true=y_true, y_pred=y_pred))
    elif (metric_name == 'precision'):
        score = np.sqrt(precision_score(y_true=y_true, y_pred=y_pred))
    elif (metric_name == 'recall'):
        score = np.sqrt(recall_score(y_true=y_true, y_pred=y_pred))
    elif (metric_name == 'roc_auc'):
        score = roc_auc_score(y_true=y_true, y_score=y_pred)
    return score


# configuration of the page
st.set_page_config(layout="wide")

SPACER = .2
ROW = 1

title_spacer1, title, title_spacer_2 = st.columns((.1, ROW, .1))
with title:
    st.title('Classification exploratory tool')
    st.markdown("""
            This app allows you to test different machine learning algorithms and combinations of preprocessing techniques 
            to solve a classification problem.
            You can choose among multiple datasets or upload your own.
            * Use the menu on the left to select ML algorithm and hyperparameters
            * The code can be accessed at [code](https://github.com/max-lutz/ML-exploration-tool).
            * Click on how to use this app to get more explanation.
            """)

title_spacer2, title_2, title_spacer_2 = st.columns((.1, ROW, .1))
with title_2:
    with st.expander("How to use this app"):
        st.markdown("""
            This app allows you to test different machine learning algorithms and combinations of preprocessing techniques.
            The menu on the left allows you to choose
            * the columns to drop (either by% of missing value or by name)
            * the transfomation to apply on your columns (imputation, scaling, encoding...)
            * the dimension reduction algorithm (none, PCA, LDA, kernel PCA)
            * the type of cross validation (KFold, StratifiedKFold)
            * the machine learning algorithm and its hyperparameters
            """)
        st.write("")
        st.markdown("""
            Each time you modify a parameter, the algorithm applies the modifications and outputs the preprocessed dataset and the results of the cross validation.
        """)


st.write("")
dataset = st.selectbox('Select dataset', ['Titanic dataset', 'Heart disease dataset', 'Iris dataset',
                                          'Diabetes dataset', 'Wine dataset', 'Load my own dataset'])
df = load_dataset(dataset)

if (df is not None):
    df, df_test = train_test_split(df, test_size=0.25, random_state=0)
    st.sidebar.header('Select feature to predict')
    _, cat_cols, _, _, _ = split_columns(df)
    target_list = [x for x in df.columns.to_list() if x in cat_cols]
    target_list.reverse()
    target_selected = st.sidebar.selectbox('Predict', target_list)

    X = df.drop(columns=target_selected)
    Y = df[target_selected].values.ravel()

    X_test = df_test.drop(columns=target_selected)
    Y_test = df_test[target_selected].values.ravel()

    # Sidebar
    # selection box for the different features
    st.sidebar.title('Preprocessing')
    st.sidebar.subheader('Dropping columns')
    missing_value_threshold_selected = st.sidebar.slider('Max missing values in feature (%)', 0, 100, 30, 1)
    cols_to_remove = st.sidebar.multiselect('Remove columns', X.columns.to_list())

    # feature with missing values
    drop_cols = cols_to_remove
    for col in X.columns:
        # put the feature in the drop trable if threshold not respected
        if ((X[col].isna().sum()/len(X)*100 > missing_value_threshold_selected) & (col not in drop_cols)):
            drop_cols.append(col)

    num_cols, cat_cols, text_cols, num_cols_missing, cat_cols_missing = split_columns(X, drop_cols)

    # create new lists for columns with missing elements
    for col in X.columns:
        if (col in num_cols and X[col].isna().sum() > 0):
            num_cols.remove(col)
            num_cols_missing.append(col)
        if (col in cat_cols and X[col].isna().sum() > 0):
            cat_cols.remove(col)
            cat_cols_missing.append(col)

    # combine text columns in one new column because countVectorizer does not accept multiple columns
    text_cols_original = text_cols
    if (len(text_cols) != 0):
        X_test['text'] = X_test[text_cols].astype(str).agg(' '.join, axis=1)
        X['text'] = X[text_cols].astype(str).agg(' '.join, axis=1)
        for cols in text_cols:
            drop_cols.append(cols)
        text_cols = "text"

    st.sidebar.subheader('Column transformation')

    categorical_imputer = wrapper_selectbox('Handling categorical missing values',
                                            ['None', 'Most frequent value', 'Constant value: "missing_value"'], len(cat_cols_missing) != 0)
    numerical_imputer = wrapper_selectbox('Handling numerical missing values',
                                          ['None', 'Median', 'Mean', 'Most frequent value', 'Constant value: -1'], len(num_cols_missing) != 0)

    encoder = wrapper_selectbox('Encoding categorical values', ['None', 'OneHotEncoder'], len(cat_cols) != 0)
    scaler = wrapper_selectbox('Scaling', ['None', 'Standard scaler',
                               'MinMax scaler', 'Robust scaler'], len(num_cols) != 0)
    text_encoder = wrapper_selectbox('Encoding text values',
                                     ['None', 'CountVectorizer', 'TfidfVectorizer'], len(text_cols) != 0)

    # need to make two preprocessing pipeline too handle the case encoding without imputer...
    preprocessing = make_column_transformer(
        (get_pipeline_missing_cat(categorical_imputer, encoder), cat_cols_missing),
        (get_pipeline_missing_num(numerical_imputer, scaler), num_cols_missing),

        (get_encoding(encoder), cat_cols),
        (get_encoding(text_encoder), text_cols),
        (get_scaling(scaler), num_cols)
    )

    st.header('Original dataset')
    row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.columns((SPACER/10, ROW*1.5, SPACER, ROW, SPACER/10))
    with row1_1:
        st.write(df)

    with row1_2:
        # display info on dataset
        st.write('Original size of the dataset', X.shape)
        st.write('Dropping ', len(drop_cols), 'feature for missing values')
        st.write('Numerical columns : ', len(num_cols))
        st.write('Categorical columns : ', len(cat_cols))
        st.write('Numerical columns with missing values : ', len(num_cols_missing))
        st.write('Categorical columns with missing values: ', len(cat_cols_missing))
        st.write('Text columns : ', len(text_cols_original))

    dim = preprocessing.fit_transform(X).shape[1]
    if ((encoder == 'OneHotEncoder') | (dim > 2)):
        dim = dim - 1

    if (dim > 2):
        st.sidebar.title('Dimension reduction')
        dimension_reduction_algorithm = st.sidebar.selectbox('Algorithm', ['None', 'Kernel PCA'])

        hyperparameters_dim_reduc = {}
        if (dimension_reduction_algorithm == 'Kernel PCA'):
            hyperparameters_dim_reduc['n_components'] = st.sidebar.slider(
                'Number of components (default = nb of features - 1)', 2, dim, dim, 1)
            hyperparameters_dim_reduc['kernel'] = st.sidebar.selectbox(
                'Kernel (default = linear)', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
    else:
        st.sidebar.title('Dimension reduction')
        dimension_reduction_algorithm = st.sidebar.selectbox('Number of features too low', ['None'])
        hyperparameters_dim_reduc = {}

    st.sidebar.title('Cross validation')
    type = st.sidebar.selectbox('Type', ['None', 'KFold', 'StratifiedKFold'])
    nb_splits = 0
    if (type is not 'None'):
        nb_splits = st.sidebar.slider('Number of splits', min_value=3, max_value=20)
    folds = get_fold(type, nb_splits)

    st.sidebar.title('Metric selection')
    metric = st.sidebar.selectbox('', ['Accuracy', 'Recall', 'Precision', 'f1', 'ROC_AUC'])

    st.sidebar.title('Model selection')
    classifier_list = ['Logistic regression', 'Support vector', 'Ridge', 'K nearest neighbors',
                       'Decision tree', 'Random forest', 'XGBoost', 'LightGBM']
    classifier = st.sidebar.selectbox('', classifier_list)

    st.sidebar.header('Hyperparameters selection')
    hyperparameters = {}
    if (classifier == 'Logistic regression'):
        hyperparameters['solver'] = st.sidebar.selectbox(
            'Solver (default = lbfgs)', ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'])
        if (hyperparameters['solver'] == 'liblinear' or hyperparameters['solver'] == 'saga'):
            hyperparameters['penalty'] = st.sidebar.selectbox('Penalty (default = l2)', ['l1', 'l2'])
        else:
            hyperparameters['penalty'] = st.sidebar.selectbox('Penalty (default = l2)', ['l2'])
        hyperparameters['C'] = st.sidebar.slider('C (default = 1.0)', 0.0, 10.0, 1.0, 0.05)

    if (classifier == 'Ridge'):
        hyperparameters['alpha'] = st.sidebar.slider('Alpha (default value = 1.0)', 0.0, 10.0, 1.0, 0.1)
        hyperparameters['solver'] = st.sidebar.selectbox(
            'Solver (default = auto)', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])

    if (classifier == 'K nearest neighbors'):
        hyperparameters['n_neighbors'] = st.sidebar.slider('Number of neighbors (default value = 5)', 1, 21, 5, 1)
        hyperparameters['metric'] = st.sidebar.selectbox('Metric (default = minkowski)', [
            'minkowski', 'euclidean', 'manhattan', 'chebyshev'])
        hyperparameters['weights'] = st.sidebar.selectbox('Weights (default = uniform)', ['uniform', 'distance'])

    if (classifier == 'Support vector'):
        hyperparameters['kernel'] = st.sidebar.selectbox('Kernel (default = rbf)', ['rbf', 'poly', 'sigmoid'])
        hyperparameters['C'] = st.sidebar.slider('C (default = 1.0)', 0.0, 10.0, 1.0, 0.05)

    if (classifier == 'Decision tree'):
        hyperparameters['criterion'] = st.sidebar.selectbox('Criterion (default = gini)', ['gini', 'entropy'])
        hyperparameters['min_samples_split'] = st.sidebar.slider('Min sample splits (default = 2)', 2, 20, 2, 1)

    if (classifier == 'Random forest'):
        hyperparameters['n_estimators'] = st.sidebar.slider('Number of estimators (default = 100)', 10, 500, 100, 10)
        hyperparameters['criterion'] = st.sidebar.selectbox('Criterion (default = gini)', ['gini', 'entropy'])
        hyperparameters['min_samples_split'] = st.sidebar.slider('Min sample splits (default = 2)', 2, 20, 2, 1)

    if (classifier == 'XGBoost'):
        hyperparameters['booster'] = st.sidebar.selectbox(
            'Algorithm (default = gbtree)', ['gbtree', 'dart', 'gblinear'])
        hyperparameters['n_estimators'] = st.sidebar.slider('Number of trees (default = 100)', 10, 500, 100, 10)
        hyperparameters['learning_rate'] = st.sidebar.slider('Learning rate (default = 0.3)', 0.01, 1.0, 0.3, 0.01)
        hyperparameters['max_depth'] = st.sidebar.slider('Maximum depth of trees (default = 6)', 0, 15, 6, 1)

    if (classifier == 'LightGBM'):
        hyperparameters['num_leaves'] = st.sidebar.slider('Number of leaves (default = 31)', 2, 100, 31, 1)
        hyperparameters['max_depth'] = st.sidebar.slider('Maximum depth (default = -1 (no limit))', -1, 200, -1, 2)
        hyperparameters['learning_rate'] = st.sidebar.slider('Learning rate (default = 0.1)', 0.01, 1.0, 0.1, 0.01)

    preprocessing_pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('dimension reduction', get_dim_reduc_algo(dimension_reduction_algorithm, hyperparameters_dim_reduc))
    ])

    pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('dimension reduction', get_dim_reduc_algo(dimension_reduction_algorithm, hyperparameters_dim_reduc)),
        ('ml', get_ml_algorithm(classifier, hyperparameters))
    ])

    preprocessing_pipeline.fit(X)
    X_preprocessed = preprocessing_pipeline.transform(X)

    st.header('Preprocessed dataset')
    display_dataframe = True
    try:
        pd.DataFrame(X_preprocessed)
    except:
        display_dataframe = False
    if (X_preprocessed.shape[1] < 100 and display_dataframe):
        st.text(f'Processed dataframe has shape: {X_preprocessed.shape}')
        st.write(X_preprocessed)
    else:
        st.text(f'Processed dataframe is too big or too sparse to display, shape: {X_preprocessed.shape}')


    row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3 = st.columns(
        (SPACER/10, ROW/2, SPACER, ROW*1.5, SPACER/10))
    try:
        with row2_1:
            cv_score = cross_validate(pipeline, X, Y, cv=folds, scoring=get_metric_name(metric), return_estimator=True)
            st.subheader('Results')
            st.write(f'Score validation [{metric}]: {round(abs(cv_score["test_score"]).mean(), 4)}')

            y_pred_test = cv_score['estimator'][0].predict(X_test)
            st.write(
                f'Score test [{metric}]: {round(abs(calculate_test_score(get_metric_name(metric), Y_test, y_pred_test)), 4)}')
        with row2_2:
            y_pred_val = cross_val_predict(pipeline, X, Y, cv=folds).round(3)
            st.write('Sampled predictions')
            df_predictions = pd.DataFrame(np.array([Y, y_pred_val]).T, columns=['Label', 'Prediction'])
            st.write(df_predictions.head(7).T)

            df_predictions = pd.DataFrame(np.array([Y_test, y_pred_test]).T, columns=['Label', 'Prediction'])
            st.write(df_predictions.head(7).T)
        
    except:
        with row2_1:
            st.subheader('Results')
            st.write(f'[ERROR] Pipeline cannot make prediction, please check that you are trying to predict the correct column.')
        with row2_2:
            st.subheader('Results')
            st.write(f'[ERROR] Pipeline cannot make prediction, please check that you are trying to predict the correct column.')


    st.subheader('Download pipeline')
    filename = 'classification.model'
    download_button_str = button.download_button(
        pipeline, filename, f'Click here to download {filename}', pickle_it=True)
    st.markdown(download_button_str, unsafe_allow_html=True)

    with st.expander('How to use the model you downloaded'):
        row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3 = st.columns((SPACER/10, ROW, SPACER, ROW, SPACER/10))

        with row2_1:
            st.write('''Put the classification.model file in your working directory
                    copy paste the code below in your notebook/code and make sure df is in the right format, 
                    with the right number of columns.
                ''')
            st.code('''
                    import joblib
                    pipeline = joblib.load('classification.model')
                    prediction = pipeline.predict(df)
                    print(prediction)
            ''')
        with row2_2:
            st.markdown('**Library versions**')
            import sklearn
            st.write("sklearn version : ", sklearn.__version__)
            st.write("numpy version : ", np.__version__)
            st.write("pandas version : ", pd.__version__)
            st.write("joblib version : ", joblib.__version__)

else:
    st.sidebar.header('')
