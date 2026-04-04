# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan
Nama : Aryo Dwi Haryanto
Email : aryodwi122@gmail.com
Id dicoding : aryo_dwi_h
## Business Understanding
Setiap kampus pasti ingin mahasiswanya lulus tepat waktu. Sayangnya, banyak mahasiswa yang terpaksa berhenti di tengah jalan (dropout atau putus kuliah). Hal ini tentu merugikan masa depan mahasiswa dan juga nama baik kampus. Melalui proyek ini, kita mencoba mencari tahu apa saja penyebab utama mahasiswa putus kuliah, sehingga kampus bisa memberikan bantuan sebelum semuanya terlambat.


### Permasalahan Bisnis

1. Angka Putus Kuliah Tinggi: Dari sekitar 4.400 data riwayat mahasiswa yang dianalisis, ada lebih dari 1.400 mahasiswa (sekitar 32%) yang berakhir putus kuliah.

2. Terlambat Menyadari: Kampus belum punya cara cepat untuk mendeteksi mahasiswa mana saja yang mulai menunjukkan tanda-tanda akan berhenti kuliah.

3. Bantuan Kurang Tepat Sasaran: Kampus kesulitan menentukan mahasiswa mana yang paling mendesak untuk diberikan pendampingan belajar atau bantuan beasiswa.
   
### Cakupan Proyek

* Menganalisis Data: Mempelajari data nilai mahasiswa, latar belakang keluarga, hingga kondisi ekonomi mereka untuk mencari pola.

* Membuat Sistem Prediksi: Menggunakan teknologi Machine Learning (komputer yang dilatih untuk belajar dari data masa lalu) agar bisa menebak nasib mahasiswa ke depannya: apakah akan Lulus, Putus Kuliah, atau Masih Berjuang (Aktif).

* Memberikan Solusi: Menyusun saran praktis untuk pihak kampus berdasarkan temuan dari data tersebut.
### Persiapan

Sumber data: data.csv
Dataset yang digunakan adalah data riwayat mahasiswa yang mencakup informasi demografi, sosial-ekonomi, dan performa akademik (semester 1 dan 2). Data ini diambil dari file data.csv yang diproses menggunakan pemisah (separator) titik koma (;).

Setup environment:
```
pip install
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib

```

## Business Dashboard

Di dalamnya, pengguna bisa melihat perbandingan jumlah mahasiswa yang lulus, aktif, dan putus kuliah. Selain itu, terdapat grafik yang menunjukkan hubungan antara nilai semester dengan status kelulusan, sehingga mempermudah manajemen dalam memantau tren penurunan performa mahasiswa secara cepat tanpa harus membaca ribuan baris data di Excel.

## Conclusion
* Proyek ini menyimpulkan bahwa nasib pendidikan mahasiswa sebenarnya bisa diprediksi lebih awal dengan melihat pola data mereka. Menggunakan model prediksi pintar, kita menemukan bahwa Semester 2 adalah masa paling kritis. Mahasiswa yang gagal menyelesaikan banyak mata kuliah atau mengalami penurunan nilai drastis di semester ini memiliki risiko sangat tinggi untuk dropout.

* Selain faktor nilai, masalah keuangan (kelancaran pembayaran UKT) juga menjadi faktor penentu apakah seorang mahasiswa akan lanjut kuliah atau berhenti. Dengan tingkat akurasi model sebesar 78%, sistem ini sudah cukup mumpuni untuk membantu kampus membedakan mana mahasiswa yang aman dan mana yang butuh bantuan segera.

### Rekomendasi Action Items (Optional)
* Aktivasi Sistem Peringatan Dini (Early Warning): Menggunakan model ini untuk memberikan notifikasi otomatis kepada dosen pembimbing jika ada mahasiswa yang nilainya anjlok di Semester 2, agar bisa segera diberikan bimbingan konseling.

* Pemantauan Mahasiswa "Abu-Abu": Memberikan perhatian ekstra kepada mahasiswa yang berstatus aktif (Enrolled) namun memiliki progres akademik yang lambat, karena mereka adalah kelompok yang paling rawan berubah status menjadi dropout.

* Evaluasi Bantuan Finansial: Mengingat faktor pembayaran UKT sangat berpengaruh, kampus sebaiknya mempermudah akses beasiswa atau skema cicilan khusus bagi mahasiswa yang terdeteksi berisiko dropout karena alasan ekonomi.

* Pendampingan Akademik Intensif: Membuat program tutorial atau belajar tambahan khusus untuk mata kuliah di semester awal yang memiliki tingkat kegagalan paling tinggi.
