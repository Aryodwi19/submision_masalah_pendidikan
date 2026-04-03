import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
st.set_page_config(
    page_title="Prediksi Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_components():
    model = joblib.load("model_rf (2).pkl")
    scaler = joblib.load("scaler (2).pkl")
    return model, scaler

model, scaler = load_components()

status_mapping = {
    0: 'Dropout',
    1: 'Enrolled',
    2: 'Graduate'
}

# ======================
# TITLE
# ======================
st.title("🎓 Prediksi Status Mahasiswa")
st.markdown("Prediksi: **Dropout | Enrolled | Graduate**")

# ======================
# INPUT USER
# ======================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Mahasiswa")

    age = st.slider("Umur", 15, 60, 20)
    gender = st.selectbox("Gender", ["Female", "Male"])
    gender = 0 if gender == "Female" else 1

    tuition = st.selectbox("SPP Lancar?", ["Ya", "Tidak"])
    tuition = 1 if tuition == "Ya" else 0

    scholarship = st.selectbox("Beasiswa?", ["Tidak", "Ya"])
    scholarship = 1 if scholarship == "Ya" else 0

with col2:
    st.subheader("Performa Akademik")

    sem1_grade = st.slider("Nilai Semester 1", 0.0, 20.0, 12.0)
    sem2_grade = st.slider("Nilai Semester 2", 0.0, 20.0, 12.0)

    approved_1 = st.slider("MK Lulus Semester 1", 0, 10, 5)
    approved_2 = st.slider("MK Lulus Semester 2", 0, 10, 5)

# ======================
# PREDIKSI
# ======================
if st.button("🔮 Prediksi"):

    input_data = pd.DataFrame([{
        'Marital_status': 1,
        'Application_mode': 1,
        'Application_order': 1,
        'Course': 171,
        'Daytime_evening_attendance': 1,
        'Previous_qualification': 1,
        'Previous_qualification_grade': 120,
        'Nacionality': 1,
        'Mothers_qualification': 1,
        'Fathers_qualification': 1,
        'Mothers_occupation': 1,
        'Fathers_occupation': 1,
        'Admission_grade': 120,
        'Displaced': 0,
        'Educational_special_needs': 0,
        'Debtor': 0,
        'Tuition_fees_up_to_date': tuition,
        'Gender': gender,
        'Scholarship_holder': scholarship,
        'Age_at_enrollment': age,
        'International': 0,
        'Curricular_units_1st_sem_credited': 0,
        'Curricular_units_1st_sem_enrolled': 6,
        'Curricular_units_1st_sem_evaluations': 6,
        'Curricular_units_1st_sem_approved': approved_1,
        'Curricular_units_1st_sem_grade': sem1_grade,
        'Curricular_units_1st_sem_without_evaluations': 0,
        'Curricular_units_2nd_sem_credited': 0,
        'Curricular_units_2nd_sem_enrolled': 6,
        'Curricular_units_2nd_sem_evaluations': 6,
        'Curricular_units_2nd_sem_approved': approved_2,
        'Curricular_units_2nd_sem_grade': sem2_grade,
        'Curricular_units_2nd_sem_without_evaluations': 0,
        'Unemployment_rate': 10.8,
        'Inflation_rate': 1.4,
        'GDP': 1.74
    }])

    try:
        with st.spinner("Memproses..."):
            scaled = scaler.transform(input_data)

            pred = model.predict(scaled)[0]
            proba = model.predict_proba(scaled)[0]

        label = status_mapping[pred]

        # ======================
        # OUTPUT
        # ======================
        st.subheader("Hasil Prediksi")

        if label == "Graduate":
            st.success(f"🎓 {label}")
        elif label == "Dropout":
            st.error(f"⚠️ {label}")
        else:
            st.warning(f"⏳ {label}")

        # ======================
        # PROBABILITAS
        # ======================
        st.subheader("Probabilitas")

        df_proba = pd.DataFrame({
            "Status": ["Dropout", "Enrolled", "Graduate"],
            "Probabilitas": proba
        })

        st.dataframe(df_proba)

        fig, ax = plt.subplots()
        ax.bar(df_proba["Status"], df_proba["Probabilitas"])
        ax.set_title("Probabilitas Prediksi")
        st.pyplot(fig)

        # ======================
        # INSIGHT
        # ======================
        st.subheader("Insight")

        if approved_2 < 3:
            st.info("Performa semester 2 rendah → risiko dropout meningkat")

        if tuition == 0:
            st.info("Tunggakan SPP berpengaruh pada dropout")

    except Exception as e:
        st.error(f"Error: {e}")