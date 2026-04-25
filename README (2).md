# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan
* Nama : Aryo Dwi Haryanto
* Email : aryodwi122@gmail.com
* Id dicoding : aryo_dwi_h
  
## Business Understanding

Institusi pendidikan tinggi menghadapi tantangan serius berupa tingginya angka mahasiswa yang tidak menyelesaikan studi mereka (dropout). Tingginya angka dropout tidak hanya merugikan mahasiswa secara individu, tetapi juga berdampak langsung pada reputasi institusi, akreditasi, dan keberlanjutan pendanaan. Tanpa sistem deteksi dini yang memadai, institusi kesulitan mengidentifikasi mahasiswa berisiko sebelum terlambat untuk melakukan intervensi.

### Permasalahan Bisnis
1. Institusi tidak memiliki sistem yang mampu mendeteksi mahasiswa berisiko dropout sejak dini, sehingga intervensi sering terlambat dilakukan.
2. Tidak diketahui faktor-faktor utama yang paling berpengaruh terhadap keputusan mahasiswa untuk berhenti kuliah.
3. Tidak ada alat prediksi berbasis data yang dapat digunakan oleh staf akademik untuk mengambil keputusan secara proaktif.
   
### Cakupan Proyek
1. Melakukan eksplorasi dan analisis data (EDA) pada dataset mahasiswa untuk menemukan pola dan faktor penyebab dropout.
2. Membangun dashboard interaktif menggunakan Looker Studio untuk memvisualisasikan temuan analisis data.
3. Membangun model machine learning (Random Forest Classifier) untuk memprediksi status mahasiswa (Graduate atau Dropout).
4. Mengembangkan aplikasi prediksi berbasis Streamlit yang dapat digunakan secara langsung oleh staf institusi.

### Persiapan

Sumber data: data_student.csv
Dataset berisi 4.424 data mahasiswa dengan 37 fitur yang mencakup informasi demografis, akademik, dan sosio-ekonomi. Tidak terdapat missing value maupun data duplikat.

## Setup environment:
Persyaratan sistem : 
```
Python 3.9.x atau lebih baru
pip 21.x+

# 1. Buat virtual environment
python -m venv venv

# 2. Aktifkan virtual environment
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install semua dependensi
pip install -r requirements.txt

# 4. Latih model (menghasilkan model_rf.pkl dan scaler.pkl)
python prediction_fix.py

# 5. Jalankan aplikasi Streamlit
streamlit run app.py

Catatan: Pastikan file data_student.csv, model_rf.pkl, dan scaler.pkl berada di direktori yang sama dengan app.py sebelum menjalankan Streamlit.

```

## Business Dashboard
Dashboard interaktif dibuat menggunakan Looker Studio untuk memvisualisasikan distribusi status mahasiswa, pola dropout berdasarkan faktor akademik dan demografis, serta perbandingan antar kelompok mahasiswa.
Dashboard dapat diakses melalui tautan berikut:Google looker studio : https://lookerstudio.google.com/reporting/010cdade-193d-42be-8ceb-093e74a7139b atau https://datastudio.google.com/s/r4i4FHdzkH4  
## Insight utama dari dashboard:
* Mahasiswa dengan status Graduate mendominasi dataset (2.209 mahasiswa), diikuti Dropout (1.421) dan Enrolled (794).
* Mahasiswa yang tidak membayar biaya kuliah tepat waktu memiliki tingkat dropout yang jauh lebih tinggi.
* Perempuan memiliki tingkat kelulusan lebih baik dibandingkan laki-laki; laki-laki menunjukkan risiko dropout yang lebih tinggi.
* Mahasiswa penerima beasiswa memiliki angka dropout yang jauh lebih rendah, menunjukkan dukungan finansial berperan sebagai faktor protektif.
* Mahasiswa yang mendaftar pada usia lebih tua cenderung memiliki risiko dropout lebih tinggi.

## Menjalankan Sistem Machine Learning
Sistem prediksi dibangun menggunakan Streamlit dan dapat dijalankan secara lokal dengan perintah berikut:
streamlit run app.py
Aplikasi akan terbuka di browser pada alamat https://submisionmasalahpendidikan-4rrkx3acmfrjhxjra46dww.streamlit.app/ 
Pengguna cukup mengisi data akademik dan demografis mahasiswa melalui form yang tersedia, lalu sistem akan memprediksi apakah mahasiswa tersebut berisiko Dropout atau berpotensi Lulus (Graduate) beserta nilai probabilitasnya.


## Conclusion
## 1. Kesimpulan Analisis Data (EDA & Dashboard)
Berdasarkan hasil eksplorasi data dan visualisasi dashboard, ditemukan faktor-faktor utama yang berkaitan erat dengan kejadian dropout:
* Performa Akademik Semester Awal adalah prediktor terkuat. Mahasiswa dengan status Dropout memiliki median nilai semester 1 paling rendah dengan rentang yang sangat lebar, bahkan banyak yang mendapat nilai 0. Kegagalan akademik di awal kuliah merupakan sinyal peringatan terbesar.
* Kondisi Finansial menjadi faktor kritis. Mahasiswa yang tidak membayar biaya kuliah tepat waktu dan yang berstatus debitur memiliki probabilitas dropout yang jauh lebih tinggi.
* Beasiswa terbukti menjadi faktor protektif. Mahasiswa penerima beasiswa memiliki angka dropout yang jauh lebih rendah.
* Gender berpengaruh terhadap risiko. Mahasiswa laki-laki menunjukkan angka dropout yang lebih tinggi dibandingkan perempuan.
* Usia saat Pendaftaran juga berperan. Mayoritas mahasiswa yang mendaftar pada usia 18–22 tahun memiliki tingkat kelulusan lebih baik. Pendaftar yang lebih tua cenderung memiliki risiko dropout lebih tinggi.

## 2. Performa Model Machine Learning
Model Random Forest Classifier dilatih menggunakan 15 fitur terpilih berdasarkan feature importance dengan hasil evaluasi sebagai berikut:
Hasil evaluasi menunjukkan performa yang sangat baik dengan akurasi keseluruhan sebesar 92,56%. Pada kelas Graduate, model memperoleh precision 0,92, recall 0,96, dan F1-score 0,94. Sementara itu, pada kelas Dropout, model mencatat precision 0,94, recall 0,87, dan F1-score 0,90. Hasil ini menunjukkan bahwa model mampu mengklasifikasikan data dengan tingkat ketepatan dan konsistensi yang tinggi.



15 Fitur Terpenting yang Digunakan Model:
1. NoFiturKeterangan1Curricular_units_2nd_sem_approved : Jumlah MK lulus semester 2
2. Curricular_units_1st_sem_approved : Jumlah MK lulus semester 1
3. Curricular_units_2nd_sem_gradeNilai : rata-rata semester 2
4. Curricular_units_1st_sem_gradeNilai : rata-rata semester 1
5. Tuition_fees_up_to_date : Status pembayaran biaya kuliah
6. Admission_grade : Nilai masuk perguruan tinggi
7. Age_at_enrollment : Usia saat mendaftar
8. Course : Program studi
9. Previous_qualification_grade : Nilai kualifikasi sebelumnya
10. Curricular_units_2nd_sem_evaluations : Jumlah evaluasi semester 2
11. Curricular_units_1st_sem_evaluations : Jumlah evaluasi semester 1
12. Fathers_occupation : Pekerjaan ayah
13. Scholarship_holder : Status penerima beasiswa
14. Gender : Jenis kelamin
15. Debtor : Status debitur / tunggakan

Interpretasi: Model sangat handal mendeteksi mahasiswa yang akan lulus (Recall Graduate = 96%). Sekitar 13% mahasiswa yang sebenarnya dropout diprediksi sebagai lulus (False Negative) kelompok ini adalah "silent dropout" yang perlu mendapat perhatian ekstra dari institusi.

## Rekomendasi Action Items
* Lakukan screening berbasis model setiap awal semester. Gunakan sistem prediksi ini untuk mengidentifikasi mahasiswa berisiko sejak semester 1, sebelum performa mereka menurun lebih jauh.
* Prioritaskan intervensi akademik. Mahasiswa dengan jumlah mata kuliah lulus semester 1 yang rendah harus segera diberikan pendampingan akademik atau program remedial.
* Perkuat program beasiswa dan bantuan finansial. Mengingat Tuition_fees_up_to_date dan Debtor menjadi faktor kunci, institusi perlu memperluas akses beasiswa dan program cicilan biaya kuliah untuk menekan risiko dropout akibat tekanan finansial.
* Buat program khusus untuk mahasiswa laki-laki dan mahasiswa usia dewasa. Kedua kelompok ini menunjukkan risiko dropout lebih tinggi dan membutuhkan pendekatan dukungan yang berbeda.
* Integrasikan dashboard dengan sistem akademik. Looker Studio dapat dikoneksikan dengan data real-time sistem akademik agar pemantauan dilakukan secara berkelanjutan, bukan hanya periodik.
