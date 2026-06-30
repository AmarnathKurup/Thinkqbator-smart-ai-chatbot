import streamlit as st

# Import functions from chatbot.py
from Health_Chat_bot import (
    getSeverityDict,
    getDescription,
    getprecautionDict,
    extract_symptoms,
    predict_disease,
    cols,
    description_list,
    precautionDictionary
)

# ------------------------------
# Load data once
# ------------------------------
getSeverityDict()
getDescription()
getprecautionDict()

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="AI HealthCare Chatbot",
    page_icon="🩺",
    layout="wide"
)

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("🩺 AI HealthCare Chatbot")

st.sidebar.info(
"""
This chatbot predicts possible diseases based on symptoms.

⚠️ It is for educational purposes only.

Always consult a qualified doctor for medical advice.
"""
)

# ------------------------------
# Main Title
# ------------------------------
st.title("🩺 AI-Driven Public Health Chatbot")

st.write(
"""
Describe your symptoms in natural language.

Example:

- I have fever and headache
- I have stomach pain and vomiting
- I have cough and runny nose
"""
)

st.divider()

# ------------------------------
# User Details
# ------------------------------

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Name")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=25
    )

with col2:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other"
        ]
    )

st.divider()

# ------------------------------
# Symptoms
# ------------------------------

symptom_text = st.text_area(
    "Describe your symptoms",
    placeholder="Example: I have fever, cough and headache."
)

# ------------------------------
# Predict Button
# ------------------------------

if st.button("🔍 Predict Disease"):

    if symptom_text.strip() == "":
        st.warning("Please enter your symptoms.")
        st.stop()

    symptoms = extract_symptoms(symptom_text, cols)

    if len(symptoms) < 2:

        st.error(
            "Please provide at least two symptoms.\n\nExample:\n\nfever, headache, cough"
        )

        st.stop()

    st.success("Detected Symptoms")

    for s in symptoms:
        st.write("✅", s.replace("_", " ").title())

    results = predict_disease(symptoms)

    top_result = results[0]

    disease = top_result["disease"]

    confidence = top_result["confidence"]

    st.divider()

    st.subheader("🩺 Prediction")

    st.metric(
        label="Most Likely Disease",
        value=disease
    )

    st.metric(
        label="Confidence",
        value=f"{float(confidence):.2f}%"
    )

    if confidence < 50:

        st.warning(
            "Prediction confidence is low. Please provide more symptoms."
        )

    st.divider()

    st.subheader("📖 Disease Description")

    st.write(
        description_list.get(
            disease,
            "No description available."
        )
    )

    st.divider()

    st.subheader("🛡️ Precautions")

    if disease in precautionDictionary:

        for p in precautionDictionary[disease]:

            if p.strip() != "":
                st.success(p)

    else:

        st.info("No precautions available.")

    st.divider()

    st.subheader("💡 General Health Tips")

    tips = [

        "Drink plenty of water.",

        "Get adequate rest.",

        "Maintain good hygiene.",

        "Eat a balanced diet.",

        "Consult a healthcare professional if symptoms worsen."

    ]

    for t in tips:

        st.write("✔️", t)

st.divider()

st.caption(
    "Developed using Machine Learning (Random Forest) + Streamlit"
)