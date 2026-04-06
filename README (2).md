# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan
* Nama : Aryo Dwi Haryanto
* Email : aryodwi122@gmail.com
* Id dicoding : aryo_dwi_h
  
## Business Understanding

Institusi pendidikan saat ini menghadapi tantangan dalam mempertahankan tingkat retensi mahasiswa. Berdasarkan data yang ada, meskipun mayoritas mahasiswa berhasil lulus (*Graduate* dengan jumlah 2.209), terdapat angka *dropout* yang cukup signifikan, yaitu sebanyak 1.421 mahasiswa (lebih dari setengah jumlah lulusan). Kondisi ini dapat berdampak buruk bagi keberlanjutan akademik dan stabilitas finansial institusi.

### Permasalahan Bisnis
Terdapat beberapa permasalahan bisnis utama yang ingin diselesaikan melalui proyek ini:
* Bagaimana cara mendeteksi mahasiswa yang memiliki risiko *dropout* secara lebih awal agar institusi dapat melakukan intervensi?
* Apa saja faktor utama (baik dari segi akademik, sosio-ekonomi, maupun demografi) yang paling berpengaruh terhadap penentuan status akhir seorang mahasiswa (Lulus atau *Dropout*)?
   
### Cakupan Proyek
Proyek ini mencakup beberapa tahapan analisis dan pemodelan data sebagai berikut:
* **Exploratory Data Analysis (EDA)**: Menganalisis distribusi data target (*Status*), distribusi variabel kategorikal seperti *gender* dan beasiswa, serta distribusi variabel numerikal seperti usia saat mendaftar.
* **Analisis Korelasi**: Memeriksa hubungan multivariat untuk melihat fitur yang berpengaruh kuat pada prediksi target, seperti korelasi antara nilai semester 1 dan semester 2.
* **Data Preparation**: Melakukan *label decoding* untuk kebutuhan *dashboard*, menyiapkan subset data untuk pemodelan biner (mengambil kategori *Dropout* dan *Graduate* saja), serta melakukan penskalaan fitur (Standard Scaler).
* **Pemodelan Machine Learning**: Membangun dan melatih model klasifikasi menggunakan algoritma *Random Forest Classifier* dengan bobot kelas yang seimbang.
* **Evaluasi & Deployment**: Mengevaluasi kinerja model menggunakan *accuracy*, *classification report* (Precision, Recall, F1-Score), serta *Confusion Matrix*, lalu mengekspor model (`model_rf.pkl`) dan skaler (`scaler.pkl`) agar siap digunakan di sistem *production*.

### Persiapan

Sumber data: data.csv
Dataset yang digunakan adalah data riwayat mahasiswa yang mencakup informasi demografi, sosial-ekonomi, dan performa akademik (semester 1 dan 2). Data ini diambil dari file data.csv yang diproses menggunakan pemisah (separator) titik koma (;).

Setup environment:
```
Sumber data: Data dimuat dari file lokal `data.csv` yang dipisahkan oleh tanda titik koma (`;`). Dataset ini terdiri dari 4.424 baris dan 37 kolom, dengan kualitas yang sangat bersih (tidak ada *missing value* maupun duplikasi).

Setup environment:

```python
# Import library untuk manipulasi data, visualisasi, dan Machine Learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

```

## Business Dashboard

Sebagai bagian dari solusi data analitik, fitur pada dataset asli telah diproses (label decoding) untuk membuat status kategori (seperti Marital_status, Course, Gender) menjadi jauh lebih mudah dibaca oleh non-teknis. Data yang telah diproses ini disimpan ke dalam file data_untuk_looker.csv agar siap digunakan sebagai sumber data di platform intelijen bisnis seperti Looker Studio. Dashboard ini akan berfokus untuk menampilkan gambaran komprehensif terkait retensi mahasiswa, pengaruh beasiswa terhadap angka kelulusan, dan peta performa akademik per kategori mahasiswa. 

* Streamlit : https://submisionmasalahpendidikan-csrnwipusaw7zjm2lgu7wq.streamlit.app/ 
* Google looket studio : https://lookerstudio.google.com/reporting/010cdade-193d-42be-8ceb-093e74a7139b 

## Conclusion
Berdasarkan analisis dan evaluasi model, dapat disimpulkan bahwa:

* Model Random Forest yang dibangun mampu mendeteksi status kelulusan (Graduate vs Dropout) dengan akurasi yang sangat baik, yaitu 89.67%.

* Model ini sangat andal dalam mendeteksi mahasiswa yang akan lulus (Recall 0.97), tetapi masih memiliki sedikit kelemahan dalam mendeteksi total mahasiswa dropout (Recall 0.81), di mana sekitar 19% dari mahasiswa dropout salah terprediksi sebagai mahasiswa lulus.

* Berdasarkan Feature Importance, performa akademik di semester 1 dan semester 2, serta status pembayaran kuliah (Tuition_fees_up_to_date) adalah faktor penentu (prediktor) terkuat atas status seorang mahasiswa dibandingkan faktor eksternal lainnya (seperti pekerjaan orang tua atau kondisi makro ekonomi).

* Rekomendasi Action Items
Berdasarkan hasil analisis, berikut adalah beberapa rekomendasi tindakan untuk pihak institusi:

* Lakukan Intervensi Dini: Institusi harus menggunakan model ini untuk mengidentifikasi mahasiswa yang memiliki probabilitas tinggi untuk dropout sejak awal semester berdasarkan nilai akademiknya. Sediakan layanan bimbingan konseling bagi mahasiswa dengan performa awal yang rendah.

* Fokus pada Bantuan Finansial: Mengingat pengaruh besar status pembayaran (Tuition_fees_up_to_date) dan beasiswa, institusi perlu menyediakan fleksibilitas pembayaran atau menambah kuota beasiswa bagi mahasiswa yang berisiko keluar akibat faktor ekonomi.

* Lakukan Penyesuaian Data Model Lanjutan: Untuk meningkatkan sensitivitas terhadap kelompok rentan (Recall Dropout yang masih 81%), tim data perlu menggunakan teknik seperti oversampling (contoh: SMOTE) guna meminimalisir bias yang terjadi akibat ketidakseimbangan data kelas target.
