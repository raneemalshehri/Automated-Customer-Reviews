import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

@st.cache_resource
def load_model():
    model_id = "tabularisai/multilingual-sentiment-analysis"  # Using the Hugging Face model ID
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Load pipeline
sentiment_analyzer = load_model()

# Streamlit UI
st.title("Sentiment Analysis with Custom Hugging Face Model")
st.write("Enter a text to analyze its sentiment:")

text_input = st.text_input("Input Text", "I’m so happy with this product!")

if text_input:
    result = sentiment_analyzer(text_input)
    st.write("**Sentiment:**", result[0]['label'])
    st.write("**Confidence:**", round(result[0]['score'] * 100, 2), "%")


