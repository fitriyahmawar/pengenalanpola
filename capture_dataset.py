import cv2
import mediapipe as mp
import numpy as np
import time
import os

FRAMES_PER_GESTURE = 80
DATASET_FILE = "dataset.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

dataset = []
if os.path.exists(DATASET_FILE):
    try:
        data = np.loadtxt(DATASET_FILE, delimiter=",")
        if data.size > 0:
            if data.ndim == 1:
                data = data.reshape(1, -1)
            dataset = data.tolist()
    except:
        dataset = []

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

recording = False
current_label = None
frame_count = 0

print("\n====================================")
print("   DATASET CAPTURE READY")
print("====================================")
print("Tekan angka 1–6 untuk mulai record gesture.")
print("Setiap gesture direkam 80 frame.")
print("Tekan Q untuk keluar.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    landmarks = None

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []
            for lm in handLms.landmark:
                lm_list.extend([lm.x, lm.y, lm.z])
            if len(lm_list) == 63:
                landmarks = lm_list
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    if recording:
        cv2.putText(frame, f"Recording Gesture {current_label}  ({frame_count}/{FRAMES_PER_GESTURE})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if landmarks is not None:
            dataset.append(landmarks + [current_label])
            frame_count += 1

        if frame_count >= FRAMES_PER_GESTURE:
            recording = False
            frame_count = 0
            np.savetxt(DATASET_FILE, np.array(dataset), delimiter=",")
            print(f"[✔] Gesture {current_label} selesai direkam & disimpan!")

    else:
        cv2.putText(frame, "Press 1-6 to Record | Press Q to Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
        current_label = int(chr(key))
        recording = True
        frame_count = 0
        print(f"[+] Mulai merekam gesture {current_label}")

    cv2.imshow("Capture Dataset", frame)

cap.release()
cv2.destroyAllWindows()

np.savetxt(DATASET_FILE, np.array(dataset), delimiter=",")
print("\n[✔] Semua dataset tersimpan ke", DATASET_FILE)
