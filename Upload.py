import pandas as pd
import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from io import StringIO

if 'new_data' in st.session_state:
    new_data = st.session_state.new_data
if 'data_column' in st.session_state:
    new_data = st.session_state.data_column
if 'labels' in st.session_state:
    new_data = st.session_state.labels

st.set_page_config(
    page_title="File Upload",
    page_icon="📎",
)
st.title('Data Upload')
upload_container = st.empty()
with upload_container.container():
    uploaded_file = st.file_uploader("Choose a file (tabular)")
    upload_btn = st.button("Upload File")
if (upload_btn):
    st.session_state.sample = pd.read_csv(uploaded_file)
    st.switch_page('pages/Data_Config.py')
