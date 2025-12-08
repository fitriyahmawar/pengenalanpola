import cv2
import mediapipe as mp
import joblib
import pygame
import os
import random
import time
import statistics

# ===========================
# LOAD MODEL + SCALER
# ===========================
knn = joblib.load("gesture_knn_model.pkl")
scaler = joblib.load("scaler.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ===========================
# MUSIC SETUP
# ===========================
MUSIC_FOLDER = os.path.expanduser("~/Music")
songs = [
    os.path.join(MUSIC_FOLDER, f)
    for f in os.listdir(MUSIC_FOLDER)
    if f.lower().endswith(".mp3")
]

if len(songs) == 0:
    raise Exception("Tidak ada file .mp3 di folder Music!")

current_index = random.randint(0, len(songs) - 1)

pygame.init()
pygame.mixer.init()

def play_song():
    pygame.mixer.music.load(songs[current_index])
    pygame.mixer.music.play()

def next_song():
    global current_index
    current_index = (current_index + 1) % len(songs)
    play_song()

def prev_song():
    global current_index
    current_index = (current_index - 1) % len(songs)
    play_song()

play_song()

gesture_text = {
    1: "PLAY",
    2: "PAUSE",
    3: "NEXT",
    4: "PREVIOUS",
    5: "VOL UP",
    6: "VOL DOWN"
}

cap = cv2.VideoCapture(0)

# ===========================
# SMOOTHING + COOLDOWN
# ===========================
pred_window = []
WINDOW_SIZE = 7
cooldown = 0.50   
last_action_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    final_pred = None

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []
            for lm in handLms.landmark:
                lm_list.extend([lm.x, lm.y, lm.z])

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            if len(lm_list) == 63:

                lm_scaled = scaler.transform([lm_list])
                pred = knn.predict(lm_scaled)[0]

                pred_window.append(pred)
                if len(pred_window) > WINDOW_SIZE:
                    pred_window.pop(0)

                try:
                    final_pred = statistics.mode(pred_window)
                except:
                    final_pred = None

            # ACTION HANDLER
            now = time.time()

            # Cooldown default 0.5 detik
            local_cooldown = cooldown

            # Cooldown gesture NEXT atau PREVIOUS
            if final_pred in [3, 4]:
                local_cooldown = 1.5

            if final_pred is not None and (now - last_action_time) >= local_cooldown:

                if final_pred == 1:
                    pygame.mixer.music.unpause()

                elif final_pred == 2:
                    pygame.mixer.music.pause()

                elif final_pred == 3:
                    next_song()

                elif final_pred == 4:
                    prev_song()

                # ==== VOLUME gesture ====
                elif final_pred == 5:
                    vol = pygame.mixer.music.get_volume()
                    pygame.mixer.music.set_volume(min(vol + 0.05, 1.0))

                elif final_pred == 6:
                    vol = pygame.mixer.music.get_volume()
                    pygame.mixer.music.set_volume(max(vol - 0.05, 0.0))

                last_action_time = now

    # ===========================
    # DISPLAY TEXT ON WEBCAM
    # ===========================
    disp = gesture_text.get(final_pred, " - ")
    cv2.putText(frame, f"Gesture: {disp}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"Volume: {int(pygame.mixer.music.get_volume() * 100)}%",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Now Playing: {os.path.basename(songs[current_index])}",
                (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Gesture Music Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
