import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from pyvi import ViTokenizer

st.title('Phân loại bài báo')

text = st.text_input('Nhập đoạn văn bản')

def tokenize(text):
    return ViTokenizer.tokenize(text)

svc_model = joblib.load('SVC.pkl')

def predict(text):
    text = tokenize(text)
    vectorizer = joblib.load('vectorizer.pkl')
    text = vectorizer.transform([text]).toarray()
    return svc_model.predict(text)[0]

if st.button('Dự đoán'):
    st.write(predict(text))


