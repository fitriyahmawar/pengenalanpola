🎵 Gesture-Based Music Control

Aplikasi ini merupakan sistem pengendalian musik berbasis gestur tangan menggunakan kamera. Sistem memanfaatkan MediaPipe untuk mendeteksi landmark tangan dan algoritma K-Nearest Neighbors (KNN) untuk mengenali jenis gesture, yang kemudian diterjemahkan menjadi perintah kontrol musik menggunakan Pygame.

🧩 Fitur Utama

1. Deteksi tangan secara real-time menggunakan kamera

2. Pengenalan gesture tangan berbasis KNN

3. Kontrol musik tanpa menyentuh perangkat
   
   (play, pause, next, previous, volume up, volume down)

4. Dataset gesture direkam sendiri oleh pengguna

5. Model dan scaler disimpan dalam bentuk file .pkl

⚙️ Instalasi

1. Python

2. opencv

3. mediapipe

4. numpy

5. scikit-learn

6. pygame

Library dapat diinstal menggunakan pip sesuai kebutuhan.

▶️ Cara Penggunaan

1. Rekam Dataset Gesture
   Jalankan program untuk merekam dataset gesture tangan
   
       python capture_dataset.py

   Instruksi penggunaan:
   
   a. Tekan angka 1–6 untuk merekam gesture
   
   b. Setiap gesture direkam sebanyak 80 frame
   
   c. Tekan Q untuk keluar
   
   d. Dataset akan otomatis tersimpan dalam file dataset.csv

2. Training Model KNN

   Setelah dataset terkumpul, lakukan proses training model

       python train_knn_model.py

    Hasil training:

    a. Model KNN tersimpan sebagai gesture_knn_model.pkl

    b. Scaler tersimpan sebagai scaler.pkl

3. Menjalankan Aplikasi Kontrol Musik

   Jalankan aplikasi utama

        python gesture_music_control.py

   Aplikasi akan:

   a. Mengaktifkan kamera

   b. Mendeteksi gesture tangan

   c. Mengontrol pemutaran musik secara real-time berdasarkan gesture

📁 Catatan

1. Musik yang digunakan berasal dari sumber lokal bawaan program

2. Tidak diperlukan folder musik tambahan

3. Pastikan kamera tidak sedang digunakan oleh aplikasi lain saat program dijalankan
