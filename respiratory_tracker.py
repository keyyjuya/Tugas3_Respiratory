"""
respiratory_tracker.py — Respiratory Tracking dari Gerak Bahu (Webcam Live)
Mata Kuliah: Sistem Teknologi Multimedia (IF25-40305) · Tugas 3

Cara menjalankan:
    python respiratory_tracker.py

Tombol Keyboard:
    q = Keluar
    m = Ganti Metode Pelacakan (Metode A: Klasik vs Metode B: Pose)
    r = Reset Buffer Sinyal / Hitung Ulang
"""

import os
import sys
import time
import urllib.request
from collections import deque

import cv2
import numpy as np
from scipy import signal
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

# --- 1. Konfigurasi Parameter Sistem ---
FS = 30.0                # Sample rate target (30 Hz)
JENDELA_DETIK = 20.0     # Buffer data sinyal (20 detik terakhir)
MINIMAL_DETIK = 8.0      # Data minimal sebelum menghitung laju napas
RPM_MIN, RPM_MAX = 6.0, 42.0  # Rentang laju napas normal (0.1 Hz - 0.7 Hz)
UPDATE_INTERVAL = 0.5   # Frekuensi pembaruan angka RPM (setiap 0.5 detik)

# --- 2. Inisialisasi Model YuNet (Metode A) & MediaPipe Pose (Metode B) ---
FOLDER = os.path.dirname(os.path.abspath(__file__))

# Model YuNet (Deteksi Wajah untuk Lokasi ROI Bahu Klasik)
MODEL_YUNET = os.path.join(FOLDER, "model", "face_detection_yunet_2023mar.onnx")
if not os.path.exists(MODEL_YUNET):
    print("⚠️ Model YuNet tidak ditemukan di folder model/. Mengunduh otomatis...")
    URL_YUNET = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    os.makedirs(os.path.dirname(MODEL_YUNET), exist_ok=True)
    urllib.request.urlretrieve(URL_YUNET, MODEL_YUNET)

detektor_yunet = cv2.FaceDetectorYN.create(MODEL_YUNET, "", (320, 320), 0.7)

# Model MediaPipe Pose Landmarker (Metode B)
MODEL_POSE = os.path.join(FOLDER, "model", "pose_landmarker_full.task")
if not os.path.exists(MODEL_POSE):
    print("⚠️ Model Pose Landmarker tidak ditemukan di folder model/. Mengunduh otomatis...")
    URL_POSE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    os.makedirs(os.path.dirname(MODEL_POSE), exist_ok=True)
    urllib.request.urlretrieve(URL_POSE, MODEL_POSE)

opsi_pose = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_POSE),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)
landmarker_pose = vision.PoseLandmarker.create_from_options(opsi_pose)


def buat_tracker_kcf():
    """Membuat Tracker KCF untuk Metode A (Klasik)."""
    for modul in (cv2, getattr(cv2, "legacy", None)):
        if modul is not None and hasattr(modul, "TrackerKCF_create"):
            return modul.TrackerKCF_create()
    if hasattr(cv2, "TrackerMIL_create"):
        return cv2.TrackerMIL_create()
    return None


# --- 3. Fungsi Preprocessing Sinyal & Estimasi Laju Napas ---
def preprocess_sinyal(waktu_list, y_list):
    """
    Melakukan alur preprocessing sinyal napas:
    1. Time Resampling -> menyamakan selang waktu ke 30 Hz
    2. Detrending -> membuang postural drift (gerakan badan pelan)
    3. Bandpass Filter -> menyaring frekuensi napas 0.1 Hz - 0.7 Hz (6-42 RPM)
    """
    t = np.asarray(waktu_list)
    y = np.asarray(y_list, dtype=float)

    # 1. Resampling ke rentang waktu seragam (30 Hz)
    t_seragam = np.arange(t[0], t[-1], 1.0 / FS)
    y_seragam = np.interp(t_seragam, t, y)

    # 2. Detrending (menghilangkan tren linear / drift)
    y_detrend = signal.detrend(y_seragam)

    # 3. Butterworth Bandpass Filter (0.1 Hz hingga 0.7 Hz)
    lowcut = RPM_MIN / 60.0  # 0.1 Hz
    highcut = RPM_MAX / 60.0 # 0.7 Hz
    b, a = signal.butter(3, [lowcut, highcut], btype="bandpass", fs=FS)
    y_filtered = signal.filtfilt(b, a, y_detrend)

    return t_seragam, y_filtered


def hitung_rpm_fft(y_filtered):
    """
    Menghitung Laju Napas (RPM) menggunakan Fast Fourier Transform (FFT).
    """
    n_fft = 4096
    spektrum = np.abs(np.fft.rfft(y_filtered * np.hanning(len(y_filtered)), n=n_fft)) ** 2
    frekuensi = np.fft.rfftfreq(n_fft, 1.0 / FS)
    rpm_axis = frekuensi * 60.0

    # Batasi pencarian puncak hanya pada rentang 6 - 42 RPM
    pita_napas = (rpm_axis >= RPM_MIN) & (rpm_axis <= RPM_MAX)
    if not np.any(pita_napas):
        return 0.0, 0.0

    indeks_puncak = np.argmax(spektrum * pita_napas)
    rpm_terestimasikan = rpm_axis[indeks_puncak]

    # Menghitung nilai keyakinan sinyal (signal confidence)
    sekitar = pita_napas & (np.abs(rpm_axis - rpm_terestimasikan) < 2.0)
    keyakinan = spektrum[sekitar].sum() / (spektrum[pita_napas].sum() + 1e-9)

    return float(rpm_terestimasikan), float(keyakinan)


# --- 4. Fungsi Visualisasi UI ---
def tulis_teks(img, teks, xy, skala=0.7, warna=(255, 255, 255), tebal=2):
    """Menulis teks dengan outline hitam agar jelas terbaca."""
    cv2.putText(img, teks, xy, cv2.FONT_HERSHEY_SIMPLEX, skala, (0, 0, 0), tebal + 3, cv2.LINE_AA)
    cv2.putText(img, teks, xy, cv2.FONT_HERSHEY_SIMPLEX, skala, warna, tebal, cv2.LINE_AA)


def gambar_grafik_sinyal(img, y_filtered, warna=(74, 195, 232)):
    """Menggambar grafik sinyal napas real-time di bagian bawah frame."""
    h, w = img.shape[:2]
    tinggi_panel = 100
    panel = img[h - tinggi_panel:h].copy()
    panel[:] = (panel * 0.3).astype(np.uint8)  # Efek semi-transparan
    img[h - tinggi_panel:h] = panel

    if len(y_filtered) > 2:
        y_cut = y_filtered[-int(12 * FS):]  # Tampilkan 12 detik terakhir
        y_norm = (y_cut - y_cut.mean()) / (np.abs(y_cut).max() + 1e-9)
        xs = np.linspace(10, w - 10, len(y_cut))
        ys = h - tinggi_panel / 2 - y_norm * (tinggi_panel / 2 - 15)
        cv2.polylines(img, [np.int32(np.stack([xs, ys], axis=1))], False, warna, 2, cv2.LINE_AA)

    tulis_teks(img, "Sinyal Gelombang Napas (12 Detik Terakhir)", (15, h - tinggi_panel + 20), 0.5, (200, 200, 200), 1)


# --- 5. Main Loop Application ---
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Gagal membuka webcam.")
        return

    metode_pelacakan = "POSE"  # Pilihan: 'KLASIK' (Metode A) atau 'POSE' (Metode B)
    
    # State Tracker Metode Klasik
    tracker_klasik = None
    roi_bahu_klasik = None

    # Buffer Data Waktu dan Posisi Vertikal Bahu
    buffer_waktu = deque()
    buffer_y_bahu = deque()

    rpm_terakhir = 0.0
    keyakinan_terakhir = 0.0
    waktu_update_terakhir = 0.0
    y_filtered_cache = np.zeros(0)

    t0 = time.time()
    nomor_frame = 0

    print("\n========================================================")
    print(" Aplikasi Respiratory Tracking dari Gerak Bahu Running!")
    print(" Tekan 'm' : Ganti Metode (Klasik / Pose)")
    print(" Tekan 'r' : Reset Data")
    print(" Tekan 'q' : Keluar")
    print("========================================================\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # Efek cermin
        h, w = frame.shape[:2]
        waktu_sekarang = time.time() - t0
        waktu_ms = int(waktu_sekarang * 1000)

        posisi_y_bahu = None

        # --- METODE B: POSE LANDMARK (MediaPipe Pose) ---
        if metode_pelacakan == "POSE":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hasil_pose = landmarker_pose.detect_for_video(mp_image, waktu_ms)

            if hasil_pose.pose_landmarks and len(hasil_pose.pose_landmarks) > 0:
                landmarks = hasil_pose.pose_landmarks[0]
                pt11 = landmarks[11]  # Bahu Kiri (Indeks 11)
                pt12 = landmarks[12]  # Bahu Kanan (Indeks 12)

                if pt11.visibility > 0.5 and pt12.visibility > 0.5:
                    x11, y11 = int(pt11.x * w), int(pt11.y * h)
                    x12, y12 = int(pt12.x * w), int(pt12.y * h)

                    # Gambar titik dan garis bahu
                    cv2.circle(frame, (x11, y11), 6, (74, 195, 232), -1)
                    cv2.circle(frame, (x12, y12), 6, (74, 195, 232), -1)
                    cv2.line(frame, (x11, y11), (x12, y12), (140, 140, 30), 2)

                    # Ambil posisi rata-rata y bahu
                    posisi_y_bahu = -((y11 + y12) / 2.0)

        # --- METODE A: KLASIK (YuNet ROI + Tracker KCF) ---
        else:
            if tracker_klasik is None:
                detektor_yunet.setInputSize((w, h))
                _, wajah = detektor_yunet.detect(frame)
                if wajah is not None and len(wajah) > 0:
                    f = max(wajah, key=lambda r: r[2] * r[3])
                    wx, wy, ww, wh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                    
                    bx = int(wx - 0.2 * ww)
                    by = int(wy + 1.2 * wh)
                    bw = int(1.4 * ww)
                    bh = int(0.5 * wh)

                    bx, by = max(0, bx), max(0, by)
                    bw, bh = min(w - bx, bw), min(h - by, bh)

                    if bw > 10 and bh > 10:
                        roi_bahu_klasik = (bx, by, bw, bh)
                        tracker_klasik = buat_tracker_kcf()
                        if tracker_klasik is not None:
                            tracker_klasik.init(frame, roi_bahu_klasik)

            else:
                ok_lacak, box_baru = tracker_klasik.update(frame)
                if ok_lacak:
                    bx, by, bw, bh = [int(v) for v in box_baru]
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (107, 158, 46), 2)
                    tulis_teks(frame, "ROI Bahu Klasik", (bx, max(15, by - 5)), 0.5, (107, 158, 46), 1)
                    
                    posisi_y_bahu = -((by + bh / 2.0))
                else:
                    tracker_klasik = None

        # --- SIMPAN POSISI Y KE BUFFER ---
        if posisi_y_bahu is not None:
            buffer_waktu.append(waktu_sekarang)
            buffer_y_bahu.append(posisi_y_bahu)

        # Buang data yang lebih tua dari JENDELA_DETIK (20 detik)
        while buffer_waktu and (waktu_sekarang - buffer_waktu[0]) > JENDELA_DETIK:
            buffer_waktu.popleft()
            buffer_y_bahu.popleft()

        # --- KALKULASI PREPROCESSING & RPM ---
        durasi_data = (buffer_waktu[-1] - buffer_waktu[0]) if len(buffer_waktu) > 1 else 0.0

        if durasi_data >= MINIMAL_DETIK:
            if waktu_sekarang - waktu_update_terakhir >= UPDATE_INTERVAL:
                t_seragam, y_filtered = preprocess_sinyal(buffer_waktu, buffer_y_bahu)
                rpm_est, conf_est = hitung_rpm_fft(y_filtered)

                rpm_terakhir = rpm_est
                keyakinan_terakhir = conf_est
                y_filtered_cache = y_filtered
                waktu_update_terakhir = waktu_sekarang

            # Tampilkan Angka Laju Napas (RPM)
            warna_rpm = (74, 195, 232) if metode_pelacakan == "POSE" else (107, 158, 46)
            tulis_teks(frame, f"Laju Napas: {rpm_terakhir:.1f} RPM", (20, 45), 1.1, warna_rpm, 3)
            tulis_teks(frame, f"Metode: {metode_pelacakan} | Keyakinan: {keyakinan_terakhir * 100:.0f}%", (20, 80), 0.6)
            
            # Gambar Grafik Sinyal
            gambar_grafik_sinyal(frame, y_filtered_cache, warna=warna_rpm)

        else:
            tulis_teks(frame, f"Mengumpulkan data napas... ({durasi_data:.0f}/{MINIMAL_DETIK:.0f}s)", (20, 45), 0.8, (0, 200, 255))
            tulis_teks(frame, "Petunjuk: Duduk tegak, bernapas secara alamiah", (20, 80), 0.55)

        # Tampilkan Informasi Bantuan Keyboard
        tulis_teks(frame, f"Tombol: 'm' ganti metode | 'r' reset | 'q' keluar", (20, h - 110), 0.45, (220, 220, 220), 1)

        cv2.imshow("Respiratory Tracking dari Gerak Bahu - IF25-40305", frame)

        tombol = cv2.waitKey(1) & 0xFF
        if tombol == ord('q'):
            break
        elif tombol == ord('m'):
            metode_pelacakan = "KLASIK" if metode_pelacakan == "POSE" else "POSE"
            tracker_klasik = None
            buffer_waktu.clear()
            buffer_y_bahu.clear()
            print(f"🔄 Berganti ke Metode: {metode_pelacakan}")
        elif tombol == ord('r'):
            tracker_klasik = None
            buffer_waktu.clear()
            buffer_y_bahu.clear()
            print("🔄 Buffer sinyal di-reset.")

        nomor_frame += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
