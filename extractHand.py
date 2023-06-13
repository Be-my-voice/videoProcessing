import cv2
import mediapipe as mp
import numpy as np
import json

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()


# class poseObj:
#     lan

class landMarkObj:
    def __init__(self, x, y ,z):
        self.x = x
        self.y = y
        self.z = z

class poseLandMarks:
    landmark = []
    def __init__(self, poseLandMarks):
        self.landmark.append(poseLandMarks)





cap = cv2.VideoCapture('evening_41.mp4')
# cap = cv2.VideoCapture(0)
print(cap.isOpened())

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_file = 'outputHands.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

width, height = 720, 720
black_screen = np.zeros((height, width, 3), dtype=np.uint8)
c = 0
while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break

    black_screen.fill(0)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(frame)

    hand_skeleton = []
    # Check if hands are detected
    if results.multi_hand_landmarks:
        print(results.multi_hand_landmarks)
        for hand_landmarks in results.multi_hand_landmarks:
            # # Extract the hand skeleton points
            
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                hand_skeleton.append((x, y))

            # Draw the hand skeleton on the frame
            
            # mp_drawing.draw_landmarks(black_screen, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # print(hand_landmarks)
            # with open('handLandmarks.json', 'a') as F:
            #     json.dump(hand_landmarks, F)    


    
    results = pose.process(frame)
    results2 = pose.process(frame)

    # for experiments
    # print(results2.pose_landmarks.landmark[0].x)
    # print(results2.pose_landmarks.landmark[0].x)
    

    # landMark = landMarkObj(results.pose_landmarks.landmark[0].x, results.pose_landmarks.landmark[0].y, results.pose_landmarks.landmark[0].z)

    # poseLandMarks = poseLandMarks(landMark)

    # for experiments
    # mp_drawing.draw_landmarks(black_screen, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    #                        mp_drawing.DrawingSpec((255, 0, 0), 2, 3),
    #                        mp_drawing.DrawingSpec((255, 0, 255), 2, 3)
    #                        )

    # real code
    mp_drawing.draw_landmarks(black_screen, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_drawing.DrawingSpec((255, 0, 0), 2, 3),
                           mp_drawing.DrawingSpec((255, 0, 255), 2, 3)
                           )

    # print(type(results.pose_landmarks))

    

    # print(results.pose_landmarks.landmark[0].x, results.pose_landmarks.landmark[0].y, results.pose_landmarks.landmark[0].z)
    # print(poseLandMarks.landMarks[0].x, poseLandMarks.landMarks[0].y, poseLandMarks.landMarks[0].z)
    # print(type(results.pose_landmarks.landmark))
    # print(len(results.pose_landmarks.landmark))

    # Display the frame
    cv2.imshow('Hand Skeletons', black_screen)

    out.write(black_screen)

    #for one frame
    # break

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()