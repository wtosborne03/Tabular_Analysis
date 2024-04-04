import pandas as pd
import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from io import StringIO

configured = False

st.set_page_config(
    page_title="Analyze",
    page_icon="📈",
)

st.title('Few-Shot Config')

new_data = st.session_state.new_data
data_column = st.session_state.data_column
labels = st.session_state.labels

st.write("Provide a ground truth for the AI to give it examples of how to classify.")

example_df = new_data.head(15)
examples = st.data_editor(example_df)


if st.button("Perform Analysis"):
    examples_list = []

    for index, example in examples.iterrows():
        ex = {"Comment": example[data_column], }
        for index, label in labels.iterrows():
            ex[label['Description']] = ' {}'.format(example[label['Label']])
        examples_list.append(ex)

    st.session_state.examples_list = examples_list
    st.switch_page('pages/Run.py')
