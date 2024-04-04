import pandas as pd
import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from io import StringIO

llm = Ollama(model="mistral:instruct")

if 'new_data' in st.session_state:
    new_data = st.session_state.new_data
if 'data_column' in st.session_state:
    new_data = st.session_state.data_column
if 'labels' in st.session_state:
    new_data = st.session_state.labels

st.set_page_config(
    page_title="Data Config",
    page_icon="📈",
)

st.title('Qualitative Data Upload/Config')
uploaded_file = st.file_uploader("Choose a file (tabular)")
if (uploaded_file is not None):
    sample_data = pd.read_csv(uploaded_file)
    st.write("Sample Data Preview: ")
    st.write(sample_data.head())
    data_column = st.selectbox(
        'Which Column would you like to analyze? (Qualitative)', sample_data.columns)

    label_df = pd.DataFrame([
        {"Label": "", "Description": ""}
    ])
    st.header("Labels")
    labels = st.data_editor(label_df, num_rows="dynamic",
                            use_container_width=True)

    if st.button("Few-Shot Examples"):
        new_data = sample_data[[data_column]]
        labels_s = labels['Label']
        for label in labels_s:
            new_data[label] = False
        st.session_state.new_data = new_data  # Save to session
        st.session_state.data_column = data_column
        st.session_state.labels = labels
        st.switch_page('pages/Few-Shot.py')
