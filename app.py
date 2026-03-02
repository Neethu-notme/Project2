import streamlit as st
import joblib
import numpy as np
from docx import Document
import io

# Load trained model & vectorizer
model = joblib.load("svm_rbf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Resume Category Predictor", layout="centered")

st.title("📄 Resume Category Predictor")
st.write("Upload a .docx resume file to predict its category.")

# File uploader
uploaded_file = st.file_uploader("Upload Resume (.docx only)", type=["docx"])

def extract_text_from_docx(file):
    doc = Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return " ".join(full_text)

if uploaded_file is not None:
    
    try:
        # Extract text
        resume_text = extract_text_from_docx(uploaded_file)
        
        if resume_text.strip() == "":
            st.warning("The uploaded file appears to be empty.")
        else:
            # Transform using saved TF-IDF
            text_vector = vectorizer.transform([resume_text])
            
            # Predict
            prediction = model.predict(text_vector)[0]
            
            # Get confidence
            probabilities = model.predict_proba(text_vector)
            confidence = np.max(probabilities) * 100
            
            st.success(f"Predicted Category: {prediction}")
            st.info(f"Confidence: {confidence:.2f}%")
    
    except Exception as e:
        st.error("Error processing file. Please upload a valid .docx file.")
