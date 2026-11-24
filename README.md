# 🫁 Tugas 3: Respiratory Tracking dari Gerak Bahu (Webcam Live)

**Mata Kuliah:** Sistem Teknologi Multimedia (IF25-40305)  
**Institut:** Institut Teknologi Sumatera (ITERA)  
**Nama:** [Kezia Adelina Tamba]  
**NIM:** [124140046]

---

## 🔗 URL Aplikasi dan Video Demonstrasi

- **Link Repositori GitHub:** [Tugas3_Respiratory](https://github.com/keyyjuya/Tugas3_Respiratory)
- **Link Video Demonstrasi:** [https://youtu.be/9jS0vsGz5Rk?si=dE8gBeQqkQQtjs6u]

---

## 💻 Panduan Penggunaan

### 1. Prasyarat dan Instalasi Library

Pastikan Python versi 3.12 sudah terinstal pada laptop. Buka Terminal atau PowerShell di dalam folder proyek, kemudian jalankan perintah berikut untuk menginstal library yang dibutuhkan:

```bash
pip install opencv-python numpy scipy mediapipe
```

### 2. Menjalankan Aplikasi

Jalankan file utama melalui terminal dengan perintah:

```bash
python respiratory_tracker.py
```

Pada saat pertama kali dijalankan, sistem akan mengunduh model AI secara otomatis ke dalam folder `model/`, yaitu:

- `face_detection_yunet_2023mar.onnx`
- `pose_landmarker_full.task`

### 3. Kontrol Keyboard

Aplikasi menyediakan beberapa kontrol keyboard untuk berinteraksi dengan sistem secara langsung.

| Tombol | Fungsi |
|:---:|---|
| `m` | Berganti metode pelacakan antara Metode A (Klasik ROI) dan Metode B (Pose Landmark). |
| `r` | Mengosongkan buffer data sinyal dan menghitung ulang laju napas dari awal. |
| `q` | Keluar dan menutup aplikasi. |

---

## 🏗️ Rancangan Pipeline Pengolahan Sinyal

### Diagram Pipeline Sistem

Sistem Respiratory Tracking memiliki empat tahapan utama, mulai dari pengambilan gambar melalui webcam hingga menghasilkan estimasi laju pernapasan secara real-time.

```text
[ Frame Webcam Live (RGB) ]
             |
             v
+--------------------------------------------------------+
| 1. FEATURE EXTRACTION (Pelacakan Bahu)                 |
|                                                        |
|    Metode A: Deteksi Wajah YuNet -> ROI Bahu -> KCF   |
|    Metode B: MediaPipe Pose -> Titik Landmark 11 & 12 |
+--------------------------------------------------------+
             |
             v
[ Sinyal Mentah Posisi Vertikal Bahu y(t) ]
             |
             v
+--------------------------------------------------------+
| 2. PREPROCESSING SINYAL TIME-SERIES                    |
|                                                        |
|    a. Time Resampling (Interpolasi Seragam 30 Hz)      |
|    b. Detrending (Menghilangkan Postural Drift)        |
|    c. Butterworth Bandpass Filter (0.1 - 0.7 Hz)       |
+--------------------------------------------------------+
             |
             v
+--------------------------------------------------------+
| 3. ESTIMASI LAJU NAPAS (DOMAIN FREKUENSI)              |
|                                                        |
|    a. Fast Fourier Transform (FFT + Hanning Window)    |
|    b. Pencarian Puncak Spektrum Dominan (6 - 42 RPM)   |
+--------------------------------------------------------+
             |
             v
+--------------------------------------------------------+
| 4. VISUALISASI REAL-TIME UI                            |
|                                                        |
|    Overlay Landmark/ROI                                |
|    Grafik Sinyal                                       |
|    Angka Estimasi Laju Napas (RPM)                     |
+--------------------------------------------------------+
```

### Penjelasan Detail dan Alasan Setiap Tahapan

#### 1. Pelacakan Posisi Vertikal Bahu (Shoulder Tracking)

Sistem mengambil posisi vertikal bahu untuk mendeteksi perubahan gerakan tubuh ketika pengguna melakukan aktivitas pernapasan.

Pada Metode B, MediaPipe Pose digunakan untuk mendeteksi dua titik landmark bahu:
- Landmark 11: Bahu kiri.
- Landmark 12: Bahu kanan.

Nilai rata-rata posisi vertikal kedua bahu dihitung menggunakan persamaan:

\[
y_{\text{bahu}}=-\frac{y_{11}+y_{12}}{2}
\]

Tanda negatif digunakan karena koordinat sumbu Y pada gambar komputer bertambah ke arah bawah. Dengan demikian, gerakan bahu ke atas menghasilkan nilai sinyal yang lebih besar.

**Alasan:** Ketika seseorang menarik napas (inhalasi), bagian dada dan bahu dapat mengalami sedikit pergerakan ke atas. Sebaliknya, ketika menghembuskan napas (ekshalasi), bahu cenderung kembali turun. Penggunaan rata-rata kedua bahu membantu mengurangi pengaruh kemiringan tubuh yang tidak seimbang.

#### 2. Time Resampling (30 Hz)

Time Resampling merupakan proses mengubah jarak waktu antar-sampel menjadi seragam menggunakan interpolasi `np.interp`.

**Alasan:** Frame rate webcam laptop tidak selalu konstan karena dipengaruhi oleh beban pemrosesan komputer. Sementara itu, perhitungan FFT membutuhkan interval waktu pengambilan sampel yang seragam.

Oleh karena itu, sinyal diinterpolasi menjadi 30 Hz atau satu sampel setiap:

\[
T=\frac{1}{30}\text{ detik}
\]

Proses ini membantu menjaga ketepatan perhitungan frekuensi dan estimasi laju napas.

#### 3. Detrending (scipy.signal.detrend)

Detrending merupakan proses menghilangkan tren atau kemiringan pada sinyal.

**Alasan:** Ketika pengguna duduk, terkadang tubuh bergerak perlahan ke depan atau ke belakang tanpa disadari. Gerakan tersebut dapat menghasilkan perubahan posisi bahu yang membentuk garis tren pada sinyal mentah.

Dengan menggunakan `scipy.signal.detrend`, tren tersebut dikurangi sehingga perubahan sinyal yang berkaitan dengan pernapasan lebih mudah dianalisis.

#### 4. Butterworth Bandpass Filter (0.1–0.7 Hz)

Butterworth Bandpass Filter digunakan untuk menyaring sinyal berdasarkan rentang frekuensi tertentu.

Sistem menggunakan rentang 0.1 Hz hingga 0.7 Hz yang setara dengan 6–42 napas per menit (RPM).

Konversinya adalah:

\[
f_{\min}=\frac{6}{60}=0.1\text{ Hz}
\]

\[
f_{\max}=\frac{42}{60}=0.7\text{ Hz}
\]

**Alasan:** Filter ini bertujuan mempertahankan komponen frekuensi yang berada dalam rentang pernapasan yang dianalisis, sekaligus mengurangi gangguan dari gerakan tubuh yang lambat dan getaran atau perubahan posisi yang lebih cepat.

Filter yang digunakan memiliki orde 3 dengan karakteristik Butterworth.

#### 5. Fast Fourier Transform (FFT + Hanning Window)

FFT merupakan metode untuk mengubah sinyal dari domain waktu menjadi domain frekuensi.

Setelah sinyal melewati proses preprocessing, sistem menerapkan Hanning Window sebelum melakukan FFT.

**Alasan:** Hanning Window digunakan untuk mengurangi efek *spectral leakage* atau kebocoran spektrum yang dapat muncul ketika sinyal dianalisis dalam durasi terbatas.

Selanjutnya, sistem mencari puncak spektrum dominan pada rentang 0.1–0.7 Hz. Frekuensi yang diperoleh dikonversi menjadi laju napas menggunakan persamaan:

\[
RPM=f_{\text{dominan}}\times60
\]

Hasilnya ditampilkan dalam satuan napas per menit (RPM).

---

## ⚙️ Parameter Penting Sistem

Berikut merupakan parameter utama yang digunakan dalam proses pengolahan sinyal:

| Parameter | Nilai / Skala | Fungsi dan Deskripsi |
|---|---|---|
| `FS` | `30.0 Hz` | Target frekuensi sampel setelah proses interpolasi seragam. |
| `JENDELA_DETIK` | `20.0 detik` | Durasi buffer sinyal terakhir yang dianalisis oleh FFT. |
| `MINIMAL_DETIK` | `8.0 detik` | Durasi minimum akumulasi sinyal sebelum estimasi RPM ditampilkan. |
| `RPM_MIN` | `6.0 RPM` | Batas bawah rentang estimasi laju napas. |
| `RPM_MAX` | `42.0 RPM` | Batas atas rentang estimasi laju napas. |
| `UPDATE_INTERVAL` | `0.5 detik` | Interval pembaruan perhitungan FFT dan estimasi RPM. |
| `Filter Order` | `3` | Orde Butterworth Bandpass IIR Filter yang digunakan. |

---

## 📊 Tabel Hasil Uji Mandiri

Pengujian dilakukan untuk melihat kemampuan sistem dalam memperkirakan laju pernapasan menggunakan dua metode pelacakan bahu yang berbeda.

| Skenario Pengujian | Pola Napas Riil | Hasil Estimasi Sistem | Tingkat Keyakinan | Status Evaluasi |
|---|---|---:|---:|---|
| Metode B (Pose) | Napas normal santai (~16 RPM) | **16.5 RPM** | 88% | ✅ Sangat akurat dan stabil |
| Metode B (Pose) | Napas cepat konsisten (~30 RPM) | **31.2 RPM** | 82% | ✅ Responsif terhadap gerakan |
| Metode A (Klasik) | Napas normal santai (~16 RPM) | **17.1 RPM** | 74% | ✅ Akurat jika baju dan background kontras |
| Metode A (Klasik) | Gerakan kepala / berpindah duduk | Unstable / Reset | 35% | ⚠️ KCF Tracker sempat kehilangan jejak |

*Catatan: Hasil pengujian merupakan hasil uji mandiri. Nilai tingkat keyakinan mengikuti indikator yang ditampilkan atau digunakan oleh sistem, bukan ukuran akurasi medis.*

---

## 🏆 Kesimpulan Metode dan Preprocessing Terbaik

### 1. Metode Pelacakan Terbaik: Metode B (MediaPipe Pose Landmark)

Metode B menggunakan MediaPipe Pose untuk mendeteksi titik persendian bahu secara langsung melalui model berbasis Deep Learning.

**Alasan pemilihan:**
- Dapat mendeteksi titik anatomi bahu kiri dan kanan.
- Tidak terlalu bergantung pada warna pakaian maupun tekstur latar belakang.
- Lebih stabil ketika terjadi sedikit perubahan posisi tubuh.
- Tidak memerlukan pelacakan objek secara manual seperti metode KCF.

Dibandingkan Metode A (YuNet + KCF Tracker), Metode B memberikan hasil pelacakan yang lebih stabil berdasarkan pengujian mandiri.

### 2. Preprocessing Paling Krusial: Resampling dan Butterworth Bandpass Filter

Kedua proses ini memiliki peran penting dalam meningkatkan kualitas sinyal sebelum dilakukan estimasi frekuensi.

- **Resampling 30 Hz:** Menyamakan interval waktu antar-sampel agar perhitungan FFT dapat dilakukan dengan lebih konsisten.
- **Butterworth Bandpass Filter (0.1–0.7 Hz):** Mempertahankan frekuensi dalam rentang pernapasan yang dianalisis dan mengurangi gangguan di luar rentang tersebut.

Kombinasi keduanya membantu menghasilkan sinyal yang lebih sesuai untuk proses estimasi laju pernapasan.

---

## ⚠️ Keterbatasan Sistem (Limitations)

Meskipun sistem dapat memperkirakan laju pernapasan secara real-time, terdapat beberapa keterbatasan yang perlu diperhatikan.

### 1. Sensitivitas terhadap Gerakan Tubuh (Gross Body Motion)

Ketika pengguna berjalan, berdiri, atau menggerakkan tangan secara aktif, perubahan posisi tubuh dapat menghasilkan sinyal yang lebih besar dibandingkan gerakan kecil akibat pernapasan. Hal ini dapat mengganggu proses estimasi laju napas.

### 2. Pernapasan Sangat Dangkal (Shallow Breathing)

Jika pengguna bernapas menggunakan diafragma atau perut tanpa menghasilkan pergerakan bahu yang cukup, perubahan posisi vertikal bahu bisa sangat kecil, bahkan kurang dari satu piksel. Akibatnya, gerakan pernapasan sulit dibedakan dari noise kamera.

### 3. Pencahayaan dan Oklusi pada Metode Klasik

Metode A (KCF Tracker) cukup sensitif terhadap perubahan pencahayaan, objek yang menutupi area bahu, serta kondisi ketika warna pakaian mirip dengan warna latar belakang. Kondisi tersebut dapat menyebabkan tracker kehilangan objek yang sedang dilacak (*lost tracking*).

---

## 🤖 Keterbukaan Bantuan AI (AI Disclosure)

Dalam pengerjaan Tugas 3 ini, teknologi AI (Gemini Notebook / LLM) digunakan sebagai asisten akademis interaktif untuk membantu beberapa tahapan pengerjaan, yaitu:

1. **Penyusunan Draf Kode Python:** Membantu memetakan integrasi antara modul *hands-on* dosen, yaitu Modul 10, 11, 12, dan 13.
2. **Debugging Kode:** Membantu mendiagnosis kesalahan struktur indeks daftar pada MediaPipe Pose (`hasil_pose.pose_landmarks`) serta penanganan *type safety* pada antrean buffer `deque`.
3. **Penyusunan Dokumentasi:** Membantu menyusun struktur dokumentasi `README.md`
4. **Pengerjaan Mandiri:** Seluruh pengujian kode menggunakan webcam secara langsung, perekaman video demonstrasi, penyesuaian parameter, serta proses *push* ke repositori GitHub dilakukan secara mandiri oleh mahasiswa.

---

**Institut Teknologi Sumatera (ITERA)**  
**Program Studi Teknik Informatika**  
**Mata Kuliah Sistem Teknologi Multimedia (IF25-40305)**