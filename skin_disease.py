# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:56:01 2026

@author: tosindataginius
"""

import streamlit as st
import numpy as np
from PIL import Image
import os

# --- CONFIGURATION ---
st.set_page_config(
    page_title="DermLens | Skin Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for a medical/modern look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007BFF;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .header-text {
        color: #1E3A8A;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_my_model():
    import tensorflow as tf
    # This ensures the app looks in the current folder for the file
    model_path = os.path.join(os.getcwd(), "my_model.keras")
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure it is uploaded to GitHub.")
        return None
        
    return tf.keras.models.load_model(model_path)

# --- CLASS MAPPING ---
# Defined based on your Confusion Matrix order
CLASS_NAMES = {
    0: "Atopic Dermatitis",
    1: "Basal Cell Carcinoma (BCC)",
    2: "Benign Keratosis-like Lesions (BKL)",
    3: "Eczema",
    4: "Melanocytic Nevi (NV)",
    5: "Melanoma",
    6: "Psoriasis, Lichen Planus and Related Diseases",
    7: "Seborrheic Keratoses and other Benign Tumors",
    8: "Tinea Ringworm Candidiasis and other Fungal Infections",
    9: "Warts Molluscum and other Viral Infections"
}

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2800/2800187.png", width=100) # Generic medical icon
    st.title("DermLens AI")
    st.info("This AI tool assists in identifying potential skin conditions using Deep Learning.")
    st.warning("Disclaimer: This is for educational purposes. Always consult a medical professional for diagnosis.")
    st.markdown("Developed by Oluwasegun Oluwatosin (tosindataginius)")
    st.link_button("Visit my LinkedIn Profile", "https://www.linkedin.com/in/oluwatosin-oluwasegun-1a9266288/")

# --- MAIN PAGE ---
st.markdown("<h1 class='header-text'>🩺 DermLens: Skin Disease Prediction</h1>", unsafe_allow_html=True)
st.write("Upload a clear image of the skin lesion for an instant AI analysis.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","jfif"])

if uploaded_file is not None:
    with st.spinner('Analyzing image...'):
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    
    # Layout with two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📸 Uploaded Image")
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True, caption="Target Area for Analysis")

    with col2:
        st.markdown("### 🔍 AI Analysis")
        
        # Preprocessing
        with st.spinner('Analyzing image... Please wait.'):
            # Match the training TARGET_SIZE
            img_processed = img.resize((224, 224)) 
            
            # EXACT same preprocessing as your training script
            #from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            img_array = np.array(img_processed)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array) # This replaces the /255.0
            

            # Prediction
            prediction = model.predict(img_array)
            predicted_class_idx = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            result = CLASS_NAMES.get(predicted_class_idx, "Unknown Class")

        # Result Display
        st.markdown(f"""
            <div class="prediction-box">
                <p style="font-size: 1.2em; color: #555;">Most Likely Condition:</p>
                <h2 style="color: #007BFF;">{result}</h2>
                <hr>
                <p style="font-size: 1.1em;">Confidence Level: <b>{confidence:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

        # Visual feedback based on confidence
        if confidence > 80:
            st.success("High confidence in this prediction.")
        elif confidence > 50:
            st.warning("Moderate confidence. Consider a clearer image.")
        else:
            st.error("Low confidence. The image may be unclear or the condition is rare.")

else:
    # Placeholder when no image is uploaded
    st.divider()
    st.info("Please upload an image file (JPG, PNG) to begin the analysis.")