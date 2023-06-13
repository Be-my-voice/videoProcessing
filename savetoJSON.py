import os
import cv2
import mediapipe as mp
import numpy as np
import json

# Create a function to extract pose from video and save it to a JSON file


def transformVideo(input_filename, output_filename, video_filename):
    # Initialize mediapipe pose solution
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Take video input for pose detection
    cap = cv2.VideoCapture(input_filename)

    # Create an empty list to store the pose landmarks
    pose_landmarks_list = []

    # Read each frame/image from the capture object
    while True:
        ret, img = cap.read()
        if not ret:
            break

        # Resize the image/frame to fit the screen
        img = cv2.resize(img, (720, 720))

        # Do pose detection
        results = pose.process(img)

        # Convert the pose landmarks to a list of dictionaries
        pose_landmarks = []
        for landmark in results.pose_landmarks.landmark:
            pose_landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })

        # Append the pose landmarks to the list
        pose_landmarks_list.append(pose_landmarks)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()

    # Write the pose landmarks and video filename to a JSON file
    with open(output_filename, 'w') as f:
        json.dump({
            'video': video_filename,
            'landmarks': pose_landmarks_list
        }, f)

    # Close all OpenCV windows
    cv2.destroyAllWindows()
    # Initialize mediapipe pose solution
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Take video input for pose detection
    cap = cv2.VideoCapture(input_filename)

    # Create an empty list to store the pose landmarks
    pose_landmarks_list = []

    # Read each frame/image from the capture object
    while True:
        ret, img = cap.read()
        if not ret:
            break

        # Resize the image/frame to fit the screen
        img = cv2.resize(img, (720, 720))

        # Do pose detection
        results = pose.process(img)

        # Convert the pose landmarks to a list of dictionaries
        pose_landmarks = []
        for landmark in results.pose_landmarks.landmark:
            pose_landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })

        # Append the pose landmarks to the list
        pose_landmarks_list.append(pose_landmarks)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()

    # Write the pose landmarks to a JSON file
    with open(output_filename, 'w') as f:
        json.dump(pose_landmarks_list, f)

    # Close all OpenCV windows
    cv2.destroyAllWindows()


# Call the function to extract pose from video and save it to a JSON file
input_folder = "Dataset/rawDataset/"
output_folder = "Dataset/processedJSONDataset/"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

transformVideo('evening_41.mp4','landmarks.json','evening_41')

# Load all files from the rawDataset folder and call transformVideo function
# for filename in os.listdir(input_folder):
#     input_file = os.path.join(input_folder, filename)
#     output_file = os.path.join(output_folder, filename.split(".")[0] + ".json")
#     print('Processing', input_file)
#     transformVideo(input_file, output_file, filename)
#     print('Done processing', input_file)
