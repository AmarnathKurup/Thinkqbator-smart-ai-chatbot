# 🩺 AI-Driven Public Health Chatbot

An AI-powered healthcare chatbot that predicts possible diseases based on user-reported symptoms using a Machine Learning model (Random Forest Classifier). The application provides disease predictions, descriptions, precautions, and general health advice through both a command-line interface and a Streamlit web application.

> **Disclaimer:** This project is developed for educational purposes only. It is **not** intended to replace professional medical advice, diagnosis, or treatment.

---

## 📌 Features

- 🔍 Disease prediction using Machine Learning
- 💬 Natural language symptom input
- 🤖 Symptom extraction with synonym support
- 📊 Top disease prediction with confidence score
- 📖 Disease description
- 🛡️ Suggested precautions
- 💡 General health tips
- 🌐 Streamlit-based web interface
- 📁 Modular project structure

---

## 🛠️ Technologies Used

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- RapidFuzz
- Joblib

---

## 📂 Project Structure

```
AI-Health-Chatbot/
│
├── app.py                     # Streamlit Web Application
├── Health_Chat_bot.py         # Chatbot Backend
├── requirements.txt
├── README.md
│
├── Data/
│   ├── Training.csv
│   └── Testing.csv
│
├── MasterData/
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   └── symptom_severity.csv
│
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Health-Chatbot.git
cd AI-Health-Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv chatbot
```

### 3. Activate the virtual environment

#### Windows

```bash
chatbot\Scripts\activate
```

#### Linux / macOS

```bash
source chatbot/bin/activate
```

### 4. Install the required packages

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

### Terminal Version

```bash
python Health_Chat_bot.py
```

### Streamlit Web Application

```bash
streamlit run app.py
```

---

## 🧠 Machine Learning Model

- Algorithm: Random Forest Classifier
- Feature Encoding: Binary Symptom Encoding
- Label Encoding: Scikit-learn LabelEncoder
- Dataset:
  - Training.csv
  - Testing.csv

---

## 📊 Dataset

The chatbot uses three datasets:

### Data

- Training.csv
- Testing.csv

### MasterData

- symptom_Description.csv
- symptom_precaution.csv
- symptom_severity.csv

These datasets are used for:

- Disease prediction
- Disease descriptions
- Severity analysis
- Recommended precautions

---

## 💻 Example

**Input**

```
I have fever, headache and cough
```

**Output**

```
Detected Symptoms
✔ High Fever
✔ Headache
✔ Cough

Possible Disease
Dengue

Confidence
72%

Precautions
✔ Drink plenty of water
✔ Take adequate rest
✔ Consult a physician
```

---

## 📌 Future Improvements

- Multiple disease prediction ranking
- Voice-based interaction
- Multilingual support
- Appointment booking
- Hospital locator
- Medical report generation
- LLM-powered conversational assistant
- Better NLP symptom extraction

---

## 👨‍💻 Developed By

**Amarnath Kurup**

M.Sc. Computer Science (Artificial Intelligence)

Digital University of Kerala

---

## 📄 License

This project is developed for academic and educational purposes.