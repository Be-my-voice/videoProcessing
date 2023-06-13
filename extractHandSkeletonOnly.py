import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils

# Create a black screen
width, height = 720, 720
black_screen = np.zeros((height, width, 3), dtype=np.uint8)

# Initialize the hand tracking module from Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Detect hands on the black screen
results = hands.process(cv2.cvtColor(black_screen, cv2.COLOR_BGR2RGB))

# Check if hands are detected
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Draw the hand skeleton on the black screen
        mp_drawing.draw_landmarks(black_screen, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# Display the black screen with the hand skeleton
cv2.imshow('Hand Skeleton', black_screen)
cv2.waitKey(0)
cv2.destroyAllWindows()
