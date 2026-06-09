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

st.markdown("### 📋 Enter Health Parameters")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.slider("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose Level", 0, 300, 120)
    blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 150, 70)
    skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)

with col2:
    insulin = st.slider("Insulin Level", 0, 900, 80)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.slider("Age", 1, 120, 25)

st.divider()

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