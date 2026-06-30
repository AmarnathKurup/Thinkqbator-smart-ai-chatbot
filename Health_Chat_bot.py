import re
import random
import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------ Load Data ------------------
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')

# Clean duplicate column names
training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
testing.columns = testing.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]
testing = testing.loc[:, ~testing.columns.duplicated()]

# Features and labels
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Encode target
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(x_train, y_train)

# ------------------ Dictionaries ------------------
severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {symptom: idx for idx, symptom in enumerate(x)}

def getDescription():
    with open('MasterData/symptom_Description.csv') as csv_file:
        for row in csv.reader(csv_file):
            description_list[row[0]] = row[1]

def getSeverityDict():
    with open('MasterData/symptom_severity.csv') as csv_file:
        for row in csv.reader(csv_file):
            try:
                severityDictionary[row[0]] = int(row[1])
            except:
                pass

def getprecautionDict():
    with open('MasterData/symptom_precaution.csv') as csv_file:
        for row in csv.reader(csv_file):
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

# ------------------ Symptom Extractor ------------------
# Predefined synonym mappings
symptom_synonyms = {
    "stomach ache": "stomach_pain",
    "belly pain": "stomach_pain",
    "tummy pain": "stomach_pain",
    "loose motion":"diarrhoea",
    "loose motions":"diarrhoea",
    "motions": "diarrhoea",
    "high temperature": "high_fever",
    "temperature": "high_fever",
    "fever": "high_fever",
    "feaver": "high_fever",
    "mild fever": "mild_fever",
    "coughing": "cough",

    "running nose":"runny_nose",
    "runny nose":"runny_nose",
    "blocked nose":"congestion",
    "throat pain": "sore_throat",
    "cold":"continuous_sneezing",
    "breathing issue": "breathlessness",
    "shortness of breath": "breathlessness",
    "body ache": "muscle_pain",

    "head pain":"headache",
    "headpain":"headache",
}

def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = user_input.lower().replace("-", " ")

    # 1. Synonym replacement
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)

    # 2. Exact match
    # Exact Match
    for symptom in all_symptoms:

        symptom_text = symptom.replace("_"," ").strip()

        if symptom_text in text:

            extracted.append(symptom)

    # 3. Fuzzy match (typo handling)
    close = get_close_matches(
            text,
            [s.replace("_"," ") for s in all_symptoms],
            n=5,
            cutoff=0.55
        )

    for item in close:

        for sym in all_symptoms:

            if sym.replace("_"," ") == item:

                extracted.append(sym)
    print("Extracted:", list(set(extracted)))

    return list(set(extracted))

# ------------------ Prediction ------------------
def predict_disease(symptoms_list):

    input_vector = np.zeros(len(symptoms_dict))

    for symptom in symptoms_list:

        if symptom in symptoms_dict:

            input_vector[symptoms_dict[symptom]] = 1

    # Convert to DataFrame so sklearn keeps feature names
    input_df = pd.DataFrame([input_vector], columns=cols)

    probabilities = model.predict_proba(input_df)[0]

    top3_idx = np.argsort(probabilities)[::-1][:3]

    results = []

    for idx in top3_idx:

        disease = le.inverse_transform([idx])[0]

        confidence = round(probabilities[idx] * 100, 2)

        results.append({
            "disease": disease,
            "confidence": confidence
        })

    return results

# ------------------ Empathy Quotes ------------------
quotes = [
    "🌸 Health is wealth, take care of yourself.",
    "💪 A healthy outside starts from the inside.",
    "☀️ Every day is a chance to get stronger and healthier.",
    "🌿 Take a deep breath, your health matters the most.",
    "🌺 Remember, self-care is not selfish."
]


# ------------------ Chatbot ------------------
def chatbot():
    getSeverityDict()
    getDescription()
    getprecautionDict()

    print("🤖 Welcome to HealthCare ChatBot")
    print("Hello! Please answer a few questions so I can understand your condition better.")

    # Basic info
    name = input("👉 What is your name? : ")
    age = input("👉 Please enter your age: ")
    gender = input("👉 What is your gender? (M/F/Other): ")

    # Initial symptoms
    symptoms_input = input("👉 Describe your symptoms in a sentence (e.g., 'I have fever and stomach pain'): ")
    symptoms_list = extract_symptoms(symptoms_input, cols)

    print("\nUser Input :", symptoms_input)
    print("Detected Symptoms :", symptoms_list)

    if len(symptoms_list) < 2:

        print("\n⚠️ Please mention at least TWO symptoms.")

        print("Example: fever, cough")

        return

    print(f"✅ Detected symptoms: {', '.join(symptoms_list)}")

    num_days = int(input("👉 For how many days have you had these symptoms? : "))
    severity_scale = int(input("👉 On a scale of 1–10, how severe do you feel your condition is? : "))
    pre_exist = input("👉 Do you have any pre-existing conditions (e.g., diabetes, hypertension)? : ")
    lifestyle = input("👉 Do you smoke, drink alcohol, or have irregular sleep? : ")
    family = input("👉 Any family history of similar illness? : ")

    # ---------------- Initial Prediction ----------------
    results = predict_disease(symptoms_list)

    top_result = results[0]
    disease = top_result["disease"]
    confidence = top_result["confidence"]
    
    if confidence < 50:
        print("\n⚠️ Low confidence prediction.")

        print("Please provide more symptoms.")

    # ---------------- Guided Questions (Disease-specific) ----------------
    print("\n🤔 Let me ask you some more questions related to", disease)
    disease_symptoms = list(training[training['prognosis'] == disease].iloc[0][:-1].index[
        training[training['prognosis'] == disease].iloc[0][:-1] == 1
    ])

    asked = 0
    for sym in disease_symptoms:
        if sym not in symptoms_list and asked < 8:
            ans = input(f"👉 Do you also have {sym.replace('_',' ')}? (yes/no): ").strip().lower()
            if ans == "yes":
                symptoms_list.append(sym)
            asked += 1


    # ---------------- Final Prediction ----------------
    results = predict_disease(symptoms_list)

    top_result = results[0]
    disease = top_result["disease"]
    confidence = top_result["confidence"]

    print("\n==============================")
    print("🩺 Possible Diseases")
    print("==============================")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['disease']} (Confidence: {result['confidence']}%)")
        
    print(f"\n📖 About {disease}")
    print(description_list.get(disease, "No description available."))

    if disease in precautionDictionary:
        print("\n🛡️ Suggested precautions:")
        for i, prec in enumerate(precautionDictionary[disease], 1):
            print(f"{i}. {prec}")

    # Random empathy quote
    print("\n💡 " + random.choice(quotes))
    print("\nThank you for using the chatbot. Wishing you good health,", name + "!")

# ------------------ Run ------------------
if __name__ == "__main__":
    chatbot()
