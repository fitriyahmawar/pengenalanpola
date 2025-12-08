import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import os

DATASET_FILE = "dataset.csv"
MODEL_FILE = "gesture_knn_model.pkl"
SCALER_FILE = "scaler.pkl"

if not os.path.exists(DATASET_FILE):
    print(f"[!] Tidak ditemukan {DATASET_FILE}. Pastikan kamu sudah capture dataset dulu.")
    sys.exit(1)

try:
    data = np.loadtxt(DATASET_FILE, delimiter=",")
except Exception as e:
    print(f"[!] Gagal load {DATASET_FILE}: {e}")
    sys.exit(1)

if data.size == 0:
    print(f"[!] {DATASET_FILE} kosong.")
    sys.exit(1)

if data.ndim == 1:
    data = data.reshape(1, -1)

X = data[:, :-1]
y = data[:, -1].astype(int)

print(f"[i] Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")

# =======================
# NORMALISASI DATA
# =======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_FILE)

# TRAIN MODEL
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_scaled, y)

# SAVE MODEL
joblib.dump(model, MODEL_FILE)

print("Training selesai!")
print("Model:", MODEL_FILE)
print("Scaler:", SCALER_FILE)
