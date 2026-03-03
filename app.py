import streamlit as st
import joblib
import numpy as np

# Load model & vectorizer
model = joblib.load("svm_rbf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Resume Classifier", layout="centered")

st.title("📄 Resume Category Predictor")

resume_text = st.text_area(
    "Input: Paste Resume Content Here",
    height=250
)

if st.button("Predict Category"):

    if resume_text.strip() == "":
        st.warning("Please paste resume content.")
    else:
        # Transform
        text_vector = vectorizer.transform([resume_text])

        # Prediction
        prediction = model.predict(text_vector)[0]

        # Confidence
        probabilities = model.predict_proba(text_vector)
        confidence = np.max(probabilities) * 100

        st.success(f"Predicted Category: {prediction}")
        st.info(f"Confidence: {confidence:.2f}%")
