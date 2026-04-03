import pandas as pd
import joblib

def predict_student_status(data_input):
    # 1. Load model dan scaler yang Anda unggah
    # Pastikan nama file sesuai dengan yang ada di komputer Anda
    try:
        model = joblib.load('model_rf (2).pkl')
        scaler = joblib.load('scaler (2).pkl')
    except FileNotFoundError:
        return "Error: File model atau scaler tidak ditemukan di direktori."

    # 2. Daftar kolom sesuai dengan urutan saat training (Total 36 kolom)
    columns = [
        'Marital_status', 'Application_mode', 'Application_order', 'Course',
        'Daytime_evening_attendance', 'Previous_qualification',
        'Previous_qualification_grade', 'Nacionality', 'Mothers_qualification',
        'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation',
        'Admission_grade', 'Displaced', 'Educational_special_needs', 'Debtor',
        'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
        'Age_at_enrollment', 'International', 'Curricular_units_1st_sem_credited',
        'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
        'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
        'Curricular_units_1st_sem_without_evaluations',
        'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
        'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
        'Curricular_units_2nd_sem_grade',
        'Curricular_units_2nd_sem_without_evaluations', 'Unemployment_rate',
        'Inflation_rate', 'GDP'
    ]

    # 3. Konversi data input menjadi DataFrame
    df_new = pd.DataFrame([data_input], columns=columns)

    # 4. Standarisasi data menggunakan scaler
    data_scaled = scaler.transform(df_new)

    # 5. Lakukan prediksi
    prediction = model.predict(data_scaled)[0]

    # 6. Mapping hasil (sesuaikan dengan label encoding di notebook Anda)
    # Biasanya: 0 = Dropout, 1 = Enrolled, 2 = Graduate
    status_map = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
    return status_map.get(prediction, "Unknown")

# --- CONTOH PENGGUNAAN ---

# Data simulasi mahasiswa baru (isi sesuai nilai fitur yang diminta)
data_mahasiswa_baru = {
    'Marital_status': 1, 'Application_mode': 1, 'Application_order': 1, 'Course': 171,
    'Daytime_evening_attendance': 1, 'Previous_qualification': 1,
    'Previous_qualification_grade': 120.0, 'Nacionality': 1, 'Mothers_qualification': 1,
    'Fathers_qualification': 1, 'Mothers_occupation': 1, 'Fathers_occupation': 1,
    'Admission_grade': 120.0, 'Displaced': 0, 'Educational_special_needs': 0, 'Debtor': 0,
    'Tuition_fees_up_to_date': 1, 'Gender': 0, 'Scholarship_holder': 0,
    'Age_at_enrollment': 20, 'International': 0, 'Curricular_units_1st_sem_credited': 0,
    'Curricular_units_1st_sem_enrolled': 6, 'Curricular_units_1st_sem_evaluations': 6,
    'Curricular_units_1st_sem_approved': 6, 'Curricular_units_1st_sem_grade': 14.5,
    'Curricular_units_1st_sem_without_evaluations': 0,
    'Curricular_units_2nd_sem_credited': 0, 'Curricular_units_2nd_sem_enrolled': 6,
    'Curricular_units_2nd_sem_evaluations': 6, 'Curricular_units_2nd_sem_approved': 6,
    'Curricular_units_2nd_sem_grade': 15.0,
    'Curricular_units_2nd_sem_without_evaluations': 0, 'Unemployment_rate': 10.0,
    'Inflation_rate': 1.5, 'GDP': 1.0
}

hasil = predict_student_status(data_mahasiswa_baru)
print(f"Hasil Prediksi Status Mahasiswa: {hasil}")