import cv2
import mediapipe as mp
import numpy as np
import json
import time

with open("landmarks.json") as file:
    data = json.load(file)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
cap = cv2.VideoCapture('sit_31.mp4')

output_file = 'outputFromJson.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, 30, (720, 720))

ret, frame = cap.read()
results = pose.process(frame)


# print(data[0][0]['x'])
# print(len(data[0]))
l = []

# for dt in data[0]:
#         results.pose_landmarks.landmark[0].x = dt['x']
#         results.pose_landmarks.landmark[0].y = dt['y']
#         results.pose_landmarks.landmark[0].z = dt['z']
        
#         # l.append((dt['x'], dt['y'], dt['z']))
#         # print(l)
#         # print(len(l))
#         print(results.pose_landmarks.landmark[0].x)

for dt1 in data:
    black_screen = np.zeros((720,720, 3), dtype=np.uint8)
    i = 0
    for dt in dt1:
        results.pose_landmarks.landmark[i].x = dt['x']
        results.pose_landmarks.landmark[i].y = dt['y']
        results.pose_landmarks.landmark[i].z = dt['z']
        i = i + 1
        # l.append((dt['x'], dt['y'], dt['z']))
        # print(l)
        # print(len(l))
    #     print(results.pose_landmarks.landmark[0].x)
    #     break
    # break

    mp_drawing.draw_landmarks(black_screen, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_drawing.DrawingSpec((255, 0, 0), 2, 3),
                           mp_drawing.DrawingSpec((255, 0, 255), 2, 3)
                           )
    
    cv2.imshow('MediaPipe Pose', black_screen)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    out.write(black_screen)

    time.sleep(0.1)

cap.release()
cv2.destroyAllWindows()
