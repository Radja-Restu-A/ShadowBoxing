# Boxing Pose Detection App

Proyek ini terdiri dari tiga bagian utama:
1. `boxing_capture_gui.py` - GUI untuk menangkap pose pengguna (terutama lengan dan bahu).
2. `boxing_train.py` - Training model deteksi pose menggunakan MediaPipe dan scikit-learn.
3. `boxing_game_api.py` - Menyediakan API berbasis FastAPI untuk merender game HTML berbasis prediksi pose.

---

## 🚀 Instalasi

1. Clone repo dan masuk ke direktori:
```bash
git clone https://github.com/nama-kamu/boxing-pose-app.git
cd boxing-pose-app
```

2. Install dependensi:
```bash
pip install -r requirements.txt
```

---

## 📸 Capture Data - `boxing_capture_gui.py`

Untuk menangkap data latih pose (lengan dan bahu):
```bash
python boxing_capture_gui.py
```
Data akan disimpan dalam file `.csv` untuk keperluan pelatihan.

---

## 🧠 Train Model - `boxing_train.py`

Melatih model klasifikasi pose:
```bash
python boxing_train.py
```
Model akan disimpan (misalnya `model.pkl`) untuk dipakai pada aplikasi utama.

---

## 🌐 API + Frontend - `boxing_game_api.py`

Menjalankan server API dan menampilkan game HTML:
```bash
uvicorn boxing_game_api:app --reload
```
Buka browser ke: `http://127.0.0.1:8000/`

---

## 📝 Catatan
- Gunakan Python 3.8+ untuk kompatibilitas terbaik dengan MediaPipe dan OpenCV.
- Pastikan webcam berfungsi jika menjalankan GUI atau game.

---

## 📂 Struktur Folder (Contoh)
```
.
├── boxing_capture_gui.py
├── boxing_train.py
├── boxing_game_api.py
├── boxing_game.html
├── model.pkl
├── requirements.txt
└── README.md
```
