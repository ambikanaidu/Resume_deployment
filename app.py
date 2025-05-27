import streamlit as st
import pickle

# Load model and vectorizer
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# Streamlit UI
st.title("Resume Classification App")
st.write("Paste your resume text below and the model will classify it into a category.")

# Text input area
resume_text = st.text_area("Enter Resume Text", height=300)

# Predict button
if st.button("Classify Resume"):
    if resume_text.strip() == "":
        st.warning("Please enter some resume text.")
    else:
        # Vectorize the input
        vectorized_text = vectorizer.transform([resume_text])
        # Predict
        prediction = model.predict(vectorized_text)[0]
        st.success(f"Predicted Resume Category: **{prediction}**")
