import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Status Mahasiswa", page_icon="🎓", layout="wide")

COURSE = {
    "Teknik Energi Terbarukan": 33,
    "Desain Komunikasi Visual (Animasi & Multimedia)": 171,
    "Kesejahteraan Sosial (Kelas Karyawan)": 8014,
    "Agroteknologi": 9003,
    "Desain Komunikasi Visual": 9070,
    "Kedokteran Hewan": 9085,
    "Teknik Informatika": 9119,
    "Peternakan": 9130,
    "Manajemen": 9147,
    "Kesejahteraan Sosial": 9238,
    "Pariwisata": 9254,
    "Ilmu Keperawatan": 9500,
    "Kedokteran Gigi": 9556,
    "Manajemen Pemasaran": 9670,
    "Ilmu Komunikasi": 9773,
    "Pendidikan Guru Sekolah Dasar (PGSD)": 9853,
    "Manajemen (Kelas Karyawan)": 9991,
}
FATHERS_OCCUPATION = {
    "Tidak Bekerja": 0,
    "Direktur / Manajer / Pimpinan": 1,
    "Dokter / Pengacara / Akuntan / Profesional Ahli": 2,
    "Teknisi / Supervisor": 3,
    "Pegawai Administrasi / Tata Usaha": 4,
    "Sales / Satpam / Customer Service": 5,
    "Petani / Nelayan / Peternak": 6,
    "Tukang Bangunan / Buruh Pabrik": 7,
    "Operator Mesin / Sopir / Montir": 8,
    "Buruh Harian / Pekerja Kasar": 9,
    "TNI / Polri": 10,
    "Lainnya": 90,
    "Tidak Ada Data": 99,
    "Dokter / Perawat / Bidan / Tenaga Medis": 122,
    "Guru / Dosen / Pengajar": 123,
    "Programmer / IT Support / Teknologi Informasi": 125,
    "Teknisi Listrik / Mesin / Sipil": 131,
    "Teknisi Laboratorium / Kesehatan": 132,
    "Staf Hukum / Sosial / Budaya": 134,
    "Staff Kantor / Admin / Operator Komputer": 141,
    "Akuntan / Keuangan / Kasir": 143,
    "Staf Administrasi Lainnya": 144,
    "Pelayan / Barber / Jasa Pribadi": 151,
    "Wirausaha / Pedagang / Pemilik Toko": 152,
    "Pengasuh / Caregiver": 153,
    "Pengrajin / Tukang Las / Tukang Kayu": 173,
    "Koki / Baker / Pengolah Makanan": 175,
    "Petugas Kebersihan / Office Boy": 191,
    "Buruh Tani / Buruh Tambak": 192,
    "Kuli Gudang / Kernet / Buruh Angkut": 193,
    "Pelayan Restoran / Waiter": 194,
}
GENDER = {"Wanita": 0, "Laki-laki": 1}
YES_NO = {"Tidak": 0, "Ya": 1}

@st.cache_resource
def load_model():
    return joblib.load("model_rf.pkl"), joblib.load("scaler.pkl")

try:
    model, scaler = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

st.title("🎓 Sistem Prediksi Status Mahasiswa")
st.markdown(
    "Masukkan data mahasiswa di bawah ini untuk memprediksi apakah mahasiswa "
    "**berisiko Dropout** atau berpotensi **Lulus (Graduate)**."
)
st.markdown("---")

if not model_loaded:
    st.error("⚠️ File `model_rf.pkl` atau `scaler.pkl` tidak ditemukan. Jalankan `python train_model.py` terlebih dahulu.")
    st.stop()

with st.form("prediction_form"):

    # ── Data Diri ─────────────────────────────────────────────────────────────
    st.markdown("#### 👤 Data Diri")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender_label      = st.selectbox("Jenis Kelamin", list(GENDER),
                              help="Jenis kelamin mahasiswa.")
        age_at_enrollment = st.number_input("Usia saat Mendaftar", 15, 70, 20,
                              help="Usia mahasiswa ketika pertama kali mendaftar.")
    with c2:
        course_label      = st.selectbox("Program Studi", list(COURSE),
                              help="Program studi yang sedang ditempuh mahasiswa.")
        fathers_occ_label = st.selectbox("Pekerjaan Ayah", list(FATHERS_OCCUPATION),
                              help="Jenis pekerjaan ayah mahasiswa.")
    with c3:
        scholarship_label = st.selectbox("Penerima Beasiswa?", list(YES_NO),
                              help="Apakah mahasiswa terdaftar sebagai penerima beasiswa?")
        tuition_label     = st.selectbox("Biaya Kuliah Sudah Lunas?",
                              ["Ya, sudah lunas", "Belum / masih ada tunggakan"],
                              help="Apakah mahasiswa sudah membayar biaya kuliah tepat waktu?")
        debtor_label      = st.selectbox("Memiliki Tunggakan / Debitur?", list(YES_NO),
                              help="Apakah mahasiswa memiliki hutang atau tunggakan ke institusi?")

    # ── Informasi Akademik ────────────────────────────────────────────────────
    st.markdown("#### 📝 Informasi Akademik")
    c4, c5 = st.columns(2)
    with c4:
        admission_grade = st.number_input("Nilai Masuk Perguruan Tinggi (0–200)",
                            0.0, 200.0, 130.0, step=0.1,
                            help="Nilai seleksi masuk perguruan tinggi.")
    with c5:
        prev_qual_grade = st.number_input("Nilai Kualifikasi Sebelumnya (0–200)",
                            0.0, 200.0, 130.0, step=0.1,
                            help="Nilai pendidikan terakhir sebelum masuk perguruan tinggi (misal: nilai UN SMA).")

    # ── Performa Akademik Semester 1 ──────────────────────────────────────────
    st.markdown("#### 📚 Performa Akademik — Semester 1")
    c6, c7, c8 = st.columns(3)
    with c6:
        cu1_approved    = st.number_input("Mata Kuliah yang Lulus (Sem 1)", 0, 26, 5,
                            help="Jumlah mata kuliah yang berhasil lulus di semester 1.")
    with c7:
        cu1_grade       = st.number_input("Nilai Rata-rata Semester 1 (0–20)", 0.0, 20.0, 13.0, step=0.1,
                            help="Rata-rata nilai seluruh mata kuliah di semester 1.")
    with c8:
        cu1_evaluations = st.number_input("Jumlah Evaluasi / Ujian (Sem 1)", 0, 45, 6,
                            help="Jumlah ujian atau evaluasi yang diikuti di semester 1.")

    # ── Performa Akademik Semester 2 ──────────────────────────────────────────
    st.markdown("#### 📊 Performa Akademik — Semester 2")
    c9, c10, c11 = st.columns(3)
    with c9:
        cu2_approved    = st.number_input("Mata Kuliah yang Lulus (Sem 2)", 0, 20, 5,
                            help="Jumlah mata kuliah yang berhasil lulus di semester 2.")
    with c10:
        cu2_grade       = st.number_input("Nilai Rata-rata Semester 2 (0-20)", 0.0, 20.0, 13.0, step=0.1,
                            help="Rata-rata nilai seluruh mata kuliah di semester 2.")
    with c11:
        cu2_evaluations = st.number_input("Jumlah Evaluasi / Ujian (Sem 2)", 0, 33, 6,
                            help="Jumlah ujian atau evaluasi yang diikuti di semester 2.")

    submit = st.form_submit_button("🔍 Prediksi Status Mahasiswa", use_container_width=True)
if submit:
    tuition_val = 1 if "Ya" in tuition_label else 0

    input_data = pd.DataFrame([{
        "Curricular_units_2nd_sem_approved":    cu2_approved,
        "Curricular_units_1st_sem_approved":    cu1_approved,
        "Curricular_units_2nd_sem_grade":       cu2_grade,
        "Curricular_units_1st_sem_grade":       cu1_grade,
        "Tuition_fees_up_to_date":              tuition_val,
        "Admission_grade":                      admission_grade,
        "Age_at_enrollment":                    age_at_enrollment,
        "Course":                               COURSE[course_label],
        "Previous_qualification_grade":         prev_qual_grade,
        "Curricular_units_2nd_sem_evaluations": cu2_evaluations,
        "Curricular_units_1st_sem_evaluations": cu1_evaluations,
        "Fathers_occupation":                   FATHERS_OCCUPATION[fathers_occ_label],
        "Scholarship_holder":                   YES_NO[scholarship_label],
        "Gender":                               GENDER[gender_label],
        "Debtor":                               YES_NO[debtor_label],
    }])

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    st.markdown("---")
    st.subheader("📈 Hasil Prediksi")

    col_a, col_b = st.columns(2)
    if prediction == 1:
        col_a.error(
            f"### ⚠️ BERISIKO DROPOUT\n\n"
            f"Probabilitas Dropout: **{probability[1]*100:.1f}%**"
        )
        st.warning(
            "**Rekomendasi:** Mahasiswa ini memerlukan perhatian lebih. "
            "Pertimbangkan untuk memberikan bimbingan akademik, konseling finansial, "
            "atau informasi program beasiswa."
        )
    else:
        col_a.success(
            f"### ✅ BERPOTENSI LULUS (Graduate)\n\n"
            f"Probabilitas Graduate: **{probability[0]*100:.1f}%**"
        )

    with col_b:
        chart_data = pd.DataFrame(
            {"Probabilitas (%)": [probability[0]*100, probability[1]*100]},
            index=["Graduate", "Dropout"]
        )
        st.bar_chart(chart_data)

    with st.expander("📋 Lihat Ringkasan Data yang Dimasukkan"):
        summary = pd.DataFrame({
        "Fitur": [
            "Jenis Kelamin", "Usia Masuk", "Program Studi",
            "Pekerjaan Ayah", "Penerima Beasiswa", "Biaya Kuliah Lunas",
            "Memiliki Tunggakan", "Nilai Masuk", "Nilai Kualifikasi Sebelumnya",
            "MK Lulus Sem 1", "Nilai Rata-rata Sem 1", "Evaluasi Sem 1",
            "MK Lulus Sem 2", "Nilai Rata-rata Sem 2", "Evaluasi Sem 2",
        ],
        "Nilai": [
            gender_label, age_at_enrollment, course_label,
            fathers_occ_label, scholarship_label,
            "Ya" if tuition_val == 1 else "Tidak",
            debtor_label, admission_grade, prev_qual_grade,
            cu1_approved, cu1_grade, cu1_evaluations,
            cu2_approved, cu2_grade, cu2_evaluations,
        ]
    })

    summary["Nilai"] = summary["Nilai"].astype(str)

    st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)
