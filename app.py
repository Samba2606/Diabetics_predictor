import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Page config
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="centered"
)

# Header
st.title("🏥 Diabetes Risk Predictor")
st.caption("Enter your health parameters below to check your diabetes risk.")
st.divider()

# Input form
st.markdown("### 📋 Enter Health Parameters")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=25)

st.divider()

# Predict button
if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        risk = probability[0][1] * 100
        st.error(f"⚠️ High Risk of Diabetes detected — {risk:.1f}% probability")
        st.markdown("""
        **Recommendations:**
        - Consult a doctor immediately
        - Monitor blood glucose regularly
        - Maintain a healthy diet and exercise routine
        """)
    else:
        safe = probability[0][0] * 100
        st.success(f"✅ Low Risk of Diabetes — {safe:.1f}% probability of being healthy")
        st.markdown("""
        **Keep it up!**
        - Maintain your healthy lifestyle
        - Regular checkups are still recommended
        - Stay active and eat well
        """)

st.divider()
st.caption("Built by Tankala Samba Siva Mani · Powered by Logistic Regression · Accuracy: 81%")