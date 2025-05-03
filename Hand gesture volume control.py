import cv2
import numpy as np
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Access system volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max = volume_ctrl.GetVolumeRange()[:2]

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Draw a small stylish volume bar
def draw_small_volume_bar(img, percent):
    x, y, w, h = 1180, 100, 20, 150
    filled = int((percent / 100) * h)

    # Background box
    cv2.rectangle(img, (x - 4, y - 4), (x + w + 4, y + h + 4), (60, 60, 60), 2)
    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), -1)

    # Fill with gradient color
    for i in range(filled):
        alpha = i / h
        color = (int(255 * (1 - alpha)), int(255 * alpha), 150)
        yi = y + h - i
        cv2.line(img, (x, yi), (x + w, yi), color, 1)

    # Draw thin border
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 255, 200), 1)

    # Display volume %
    if percent > 0:
        cv2.putText(img, f"{int(percent)}%", (x - 60, y + h + 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 180), 2)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    lm_list = []
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        for id, lm in enumerate(hand.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append((cx, cy))
        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

        if lm_list:
            x1, y1 = lm_list[4]
            x2, y2 = lm_list[8]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            distance = math.hypot(x2 - x1, y2 - y1)
            vol = np.interp(distance, [30, 200], [vol_min, vol_max])
            vol_percent = np.interp(distance, [30, 200], [0, 100])
            volume_ctrl.SetMasterVolumeLevel(vol, None)

            # Draw small finger indicators
            cv2.circle(img, (x1, y1), 10, (255, 200, 100), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (200, 255, 100), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (180, 255, 100), 3)
            cv2.circle(img, (cx, cy), 8, (255, 100, 200), cv2.FILLED)

            draw_small_volume_bar(img, vol_percent)

    cv2.imshow("🎚️ Mini Gesture Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
