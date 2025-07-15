import streamlit as st
import requests
import re
import spacy
from bs4 import BeautifulSoup
from transformers import pipeline
import torch
nlp = spacy.load("en_core_web_sm")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def scrape_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        p_char = soup.find_all('p')
        paragraph = '.'.join(p.get_text() for p in p_char)
        cleaned_text = re.sub(r'\s+|[^A-Za-z0-9.,]+', ' ', paragraph).strip()
        doc = nlp(cleaned_text)
        final_text = [token.lemma_ for token in doc if not token.is_punct]
        return ' '.join(final_text)
    else:
        return None

def answer_question(context, question):
    try:
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        return f"Error answering question: {str(e)}"

# Streamlit UI
st.title("Scrapy: The Chatbot:robot:")
st.markdown("Enter a URL and ask a question based on its content. Let the bot do some web linguistics magic 🧙‍♂️")

url_input = st.text_input("Enter URL:")
q_input = st.text_input("Enter any question:")

if st.button("Click 🐱"):
    if url_input and q_input:
        content = scrape_data(url_input)
        if content:
            ans = answer_question(content, q_input)
            st.success(f"Fetched Answer: \n\n{ans}")
        else:
            st.error("Could not scrape the webpage. Please check the URL.")
    else:
        st.warning("Please enter both a URL and a question.")
