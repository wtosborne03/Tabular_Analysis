import pandas as pd
import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from io import StringIO
import re
import numpy as np


llm = Ollama(model="mistral")

st.set_page_config(
    page_title="Run",
    page_icon="📈",
)

st.title('Run')

new_data = st.session_state.new_data
data_column = st.session_state.data_column
labels = st.session_state.labels
examples_list = st.session_state.examples_list


def string_to_booleans(input_string):
    matches = re.findall(r'True|False|Unknown', input_string)
    boolean_list = [True if match == 'True' else False for match in matches]

    return boolean_list


if st.button("Analyze"):
    qtemp = "Comment: '{Comment}'  \nAnalysis:"
    for index, label in labels.iterrows():
        qtemp += "  \n" + label['Description'] + \
            ": {" + label['Description'] + "}"

    labels_s = labels['Label'].tolist()
    labels_s.append("Comment")

    example_prompt = PromptTemplate(
        input_variables=labels_s, template=qtemp
    )

    prompt = FewShotPromptTemplate(
        examples=examples_list,
        example_prompt=example_prompt,
        prefix="Instructions: Your name is John. You need to classify data like how these are examples are laid out. Continue classifying the next examples. Do not explain your reasoning, just follow the structure of the comments before the new one. ONLY respond with 'True' or 'False'. If the answer is unknown, use your judgement and only classify it as 'True' or 'False'. \n###  \nExamples:",
        suffix="\n###  \nJohn:  \nComment: '{comment}'  \nAnalysis:",
        input_variables=["comment"],
    )

    print(prompt.format(comment='[INPUT]'))

    # Perform Analysis
    batch_size = 5  # Adjust batch size according to your requirements and limitations
    batches = [new_data[i:i + batch_size]
               for i in range(0, new_data.shape[0], batch_size)]

    for batch in batches:
        # Create a batched prompt for processing
        batch_prompts = [prompt.format(comment=row[data_column])
                         for index, row in batch.iterrows()]

        # Process the batch
        try:
            batch_results = llm.batch(batch_prompts)
            for index, result in zip(batch.index, batch_results):
                results = string_to_booleans(result)
                i = 0
                try:
                    for label in labels['Label']:
                        new_data.loc[index, label] = results[i]
                        i += 1
                    st.write(new_data.loc[index])
                except Exception as e:
                    st.write(f'Error analyzing row {index}: {e}')
        except Exception as e:
            st.write(f'Error processing batch: {e}')
