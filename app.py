import streamlit as st
from multiapp import MultiApp
from apps import home, classification, classification_summary, regression, regression_summary # import your app modules here

#configuration of the page
st.set_page_config(layout="wide")

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Classification", classification.app)
app.add_app("Classification summary", classification_summary.app)
app.add_app("Regression", regression.app)
app.add_app("Regression summary", regression_summary.app)

# The main app
app.run()