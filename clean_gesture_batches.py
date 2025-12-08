import pandas as pd
import numpy as np

FILE = "dataset.csv"
BATCH_SIZE = 80  # jumlah frame per batch

print("\n=== ADVANCED DATASET CLEANER ===\n")

# Baca dataset
df = pd.read_csv(FILE, header=None)

# Kolom terakhir adalah label gesture
labels = df.iloc[:, -1].values

# Hitung total batch per gesture (1–6)
gesture_batches = {}

print("Total batch per gesture:")
for g in range(1, 7):
    count_rows = np.sum(labels == g)
    count_batches = count_rows // BATCH_SIZE
    gesture_batches[g] = count_batches
    print(f"Gesture {g}: {count_batches} batch")

print("\n===============================\n")

# Pilih gesture mana yang mau dibersihkan
gesture = int(input("Hapus batch dari gesture berapa? (1–6): "))

if gesture not in gesture_batches:
    print("[!] Gesture tidak valid.")
    exit()

total_batch_for_gesture = gesture_batches[gesture]
print(f"Gesture {gesture} memiliki total {total_batch_for_gesture} batch.")

# Berapa batch yang mau dihapus
to_delete = int(input(f"Mau hapus berapa batch terakhir dari gesture {gesture}? "))

if to_delete < 1 or to_delete > total_batch_for_gesture:
    print("[!] Jumlah batch tidak valid.")
    exit()

print(f"\n[!] Menghapus {to_delete} batch terakhir dari gesture {gesture}...\n")

# Ambil index seluruh baris label gesture tsb
indices = np.where(labels == gesture)[0]

# Batch per gesture selalu berurutan, jadi cukup ambil baris terakhir
rows_to_delete = to_delete * BATCH_SIZE

delete_start = len(indices) - rows_to_delete
delete_end = len(indices)

rows_idx = indices[delete_start:delete_end]

print(f"Menghapus total {len(rows_idx)} baris:")
print(f"Index mulai: {rows_idx[0]}")
print(f"Index akhir : {rows_idx[-1]}")

# Hapus baris
df_clean = df.drop(rows_idx)

# Simpan
df_clean.to_csv(FILE, index=False, header=False)

print("\n[✓] Batch gesture berhasil dihapus!")
print(f"[✓] Total baris sekarang: {len(df_clean)}")
