# Tugas 3: Respiratory Tracking dari Gerak Bahu (Webcam Live)
**Mata Kuliah**: Sistem Teknologi Multimedia (IF25-40305)  
**Nama**: [Kezia Adelina Tamba]  
**NIM**: [124140046]  

---

## 📹 Link Video Demonstrasi
- **URL Video Demo**: [Tempelkan Link YouTube / Google Drive Video Kamu di Sini]

---

## 📌 Deskripsi Program
Aplikasi ini melakukan pelacakan laju napas (*respiratory rate*) secara *real-time* berbasis gerak vertikal bahu menggunakan webcam. Ketika seseorang bernapas (inhalasi dan ekshalasi), gerakan naik-turun bahu diekstrak menjadi sinyal time-series, dibersihkan melalui alur *preprocessing* sinyal, dan dihitung laju napasnya dalam satuan **Respirations Per Minute (RPM)** menggunakan Fast Fourier Transform (FFT).

Aplikasi menyediakan 2 pilihan metode pelacakan:
1. **Metode A (Klasik)**: Menggunakan detektor wajah **YuNet** untuk membuat ROI bahu di bawah wajah, lalu melacak pergeseran piksel bahu dengan **Tracker KCF**.
2. **Metode B (Pose Landmark)**: Menggunakan **MediaPipe Pose Landmarker** untuk mengunci titik kerangka anatomi **titik 11 (bahu kiri)** dan **titik 12 (bahu kanan)** secara otomatis.

---

## 🏗️ Arsitektur Pipeline Pengolahan Sinyal

```text
[ Webcam Frame ] ──> [ Feature Tracking ] ──> [ Sinyal Mentah y(t) ]
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
   Metode A (Klasik)                Metode B (Pose)
   (YuNet ROI + KCF Tracker)        (MediaPipe titik 11 & 12)
                            │
                            ▼
                    [ Preprocessing ]
             1. Resampling Time-Series (30 Hz)
             2. Detrending (Membuang Postural Drift)
             3. Butterworth Bandpass Filter (0.1–0.7 Hz)
                            │
                            ▼
                  [ Estimasi Laju Napas ]
                     (FFT Spectrum Peak)
                            │
                            ▼
                   [ Visualisasi UI ]
          (Overlay ROI/Landmark + Sinyal Real-time + RPM)