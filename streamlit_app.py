import streamlit as st


import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# --------------------
# CONFIG & TITLE
# --------------------
st.set_page_config(page_title="Paws & Predicts", layout="centered")

# --------------------
# LOAD MODEL
# --------------------
@st.cache_resource
def load_model():
    return joblib.load("xgb_model_realistic.pkl")

model = load_model()



# --------------------
# LOGO + HEADER
# --------------------
if os.path.exists("logo.png"):
    st.image("logo.png", width=400)
st.title("🐾 Paws & Predicts: Pet Adoption Speed Predictor")

# --------------------
# HORIZONTAL TABS
# --------------------
tab1, tab2, tab3 = st.tabs(["🐶 Predict", "📊 EDA", "🧪 Model Results"])

# --------------------
# PREDICT TAB
# --------------------
with tab1:
    st.header("📋 Enter Pet Details")

    with st.form("predict_form"):
        breed = st.selectbox("Breed", [
            "Labrador", "Poodle", "German Shepherd", "Bulldog", "Mixed", 
            "Beagle", "Golden Retriever", "Australian Shepherd", "Husky"
        ])
        age = st.slider("Age in months", 1, 120, 12)
        size = st.selectbox("Size", ["Tiny", "Small", "Medium", "Large", "XL"])
        sterilized = st.radio("Sterilized", ["Yes", "No"])
        health = st.selectbox("Health", ["Healthy", "Minor Injury", "Serious Injury"])
        temperament = st.multiselect("Select 3 temperament traits", [
            "Playful", "Shy", "Calm", "Aggressive", "Friendly", 
            "Independent", "Anxious", "Affectionate"
        ])
        good_with_kids = st.radio("Good with Children?", ["Yes", "No"])
        good_with_pets = st.radio("Good with Other Pets?", ["Yes", "No"])
        photo_count = st.slider("Photo Count", 0, 5, 3)
        desc_length = st.slider("Description Length", 20, 500, 150)

        submitted = st.form_submit_button("Predict Adoption Speed")

    if submitted:
        if len(temperament) != 3:
            st.warning("Please select exactly 3 temperament traits.")
        else:
            # Format input
            data = {
                "Breed": breed,
                "AgeInMonths": age,
                "Size": size,
                "Sterilized": sterilized,
                "Health": health,
                "GoodWithChildren": good_with_kids,
                "GoodWithOtherPets": good_with_pets,
                "PhotoCount": photo_count,
                "DescriptionLength": desc_length,
            }

            # Add temperament flags
            for trait in ["Playful", "Shy", "Calm", "Aggressive", "Friendly", "Independent", "Anxious", "Affectionate"]:
                data[trait] = 1 if trait in temperament else 0

            input_df = pd.DataFrame([data])

            # Predict
            pred = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0]

            label_map = {
                0: "Not Adopted 😢",
                1: "Slow Adoption",
                2: "Medium Speed",
                3: "Fast Adoption 🐕‍🦺",
                4: "Very Fast Adoption 🚀"
            }

            st.subheader("🐶 Prediction Result:")
            st.success(f"**{label_map[pred]}**")
            st.markdown(f"Confidence: `{pred_proba[pred]:.2%}`")

# --------------------
# EDA TAB
# --------------------
with tab2:
    st.header("📊 EDA Insights")
    for i in range(1, 7):
        plot_path = f"eda_plot_{i}.png"
        if os.path.exists(plot_path):
            st.image(plot_path, use_column_width=True)
        else:
            st.info(f"{plot_path} not found.")

# --------------------
# MODEL RESULTS TAB
# --------------------
with tab3:
    st.header("🧪 Model Performance")
    if os.path.exists("classificationreport.png"):
        st.image("classificationreport.png")
    st.markdown("""
    **Model**: XGBoostClassifier  
    **Features**: Breed, size, sterilization, health, age, temperament traits, photo count, and description length  
    **Data**: 1000 synthetic pets with realistic adoption behavior patterns  
    **Macro F1 Score**: ~0.24 before enhancements  
    """)
