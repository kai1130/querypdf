import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

import pandas as pd
from pdfminer.high_level import extract_text

def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    
    return knowledgeBase

def query_document(query):
    docs = knowledgeBase.similarity_search(query)

    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type='stuff')

    with get_openai_callback() as cost:
        response = chain.run(input_documents=docs, question=query)
        st.write(cost)
        st.write(f'Query: {query}')
        st.write(f'Response: {response}')
    
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

st.title('Query PDF')
st.text('natural language chat interface to ask your pdf files some questions')

path = "./2021-tesla-impact-report.pdf"
if 'df' not in st.session_state:
    st.session_state['filetext'] = extract_text(path)

st.divider()
st.subheader('Select Data Source')
st.caption('Default PDF = Tesla 2021 Impact Report')

upload = st.file_uploader("Upload Custom Dataset (.pdf file)")
if upload:
    st.session_state['filetext'] = extract_text(upload)

st.text_area(st.session_state['filetext'])

st.divider()
st.subheader('Chat Interface')

with st.form('chat_area'):
    text = st.text_area('Enter Question(s)', 'what are the different corporate governance committees?\nwhat is the 10th top choice company for graduate engineers?')
    submitted = st.form_submit_button('Submit Question')

    if submitted:

        query_document(query)