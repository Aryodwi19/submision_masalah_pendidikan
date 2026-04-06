import streamlit as st
import pandas as pd
import joblib
import os

# ===============================
# 1. PAGE CONFIG & STYLING
# ===============================
st.set_page_config(page_title="Student Success Predictor", layout="centered")

st.title("🎓 Student Status Prediction")
st.markdown("Enter the student's details below to predict if they are likely to **Graduate** or **Dropout**.")

# ===============================
# 2. LOAD ASSETS (Cached)
# ===============================
@st.cache_resource
def load_assets():
    model = joblib.load("model_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    # Get feature order from scaler
    try:
        feature_order = scaler.feature_names_in_.tolist()
    except AttributeError:
        # Fallback if names aren't in the scaler
        feature_order = [
            "Age", "Gender", "Scholarship_holder", "Tuition_fees_up_to_date",
            "Curricular_units_1st_sem_approved", "Curricular_units_2nd_sem_approved"
        ]
    return model, scaler, feature_order

# Load model, scaler, and features
try:
    model, scaler, feature_order = load_assets()
except Exception as e:
    st.error(f"Could not load model files. Error: {e}")
    st.stop()

# ===============================
# 3. USER INPUT UI
# ===============================
with st.form("prediction_form"):
    st.header("Student Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age at Enrollment", min_value=15, max_value=100, value=20)
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        scholarship = st.selectbox("Scholarship Holder?", options=["Yes", "No"])
    
    with col2:
        tuition = st.selectbox("Tuition Fees Up to Date?", options=["Yes", "No"])
        units_1st = st.number_input("1st Sem Approved Units", min_value=0, max_value=30, value=5)
        units_2nd = st.number_input("2nd Sem Approved Units", min_value=0, max_value=30, value=5)

    submit = st.form_submit_button("Predict Result")

# ===============================
# 4. PREDICTION LOGIC
# ===============================
if submit:
    # Prepare Input Dictionary
    input_data = {
        "Age": age,
        "Gender": gender,
        "Scholarship_holder": 1 if scholarship == "Yes" else 0,
        "Tuition_fees_up_to_date": 1 if tuition == "Yes" else 0,
        "Curricular_units_1st_sem_approved": units_1st,
        "Curricular_units_2nd_sem_approved": units_2nd
    }
    
    # Create DataFrame and align with feature_order
    df = pd.DataFrame([input_data])
    
    # Fill missing features with 0 (if any) and reorder
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_order]

    # Scaling & Prediction
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]
    probabilities = model.predict_proba(df_scaled)[0]

    # ===============================
    # 5. DISPLAY RESULTS
    # ===============================
    st.divider()
    
    if prediction == 1:
        st.success(f"### Predicted Result: GRADUATE")
    else:
        st.error(f"### Predicted Result: DROPOUT")

    # Show confidence scores
    col_drop, col_grad = st.columns(2)
    col_drop.metric("Dropout Probability", f"{round(probabilities[0]*100, 2)}%")
    col_grad.metric("Graduate Probability", f"{round(probabilities[1]*100, 2)}%")