import streamlit as st
from multiapp import MultiApp
from apps import classification, regression # import your app modules here

#configuration of the page
st.set_page_config(layout="wide")

app = MultiApp()

# Add all your application here
app.add_app("Classification", classification.app)
app.add_app("Regression", regression.app)

# The main app
app.run()