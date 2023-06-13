import os
import cv2
import mediapipe as mp
import numpy as np
import json

# Create a function to extract pose from video and save it to a JSON file


# Create a function to extract pose from JSON and save it to a video file
def transformVideo(input_filename, output_filename):
    # Initialize mediapipe pose solution
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize mediapipe drawing utils
    mp_drawing = mp.solutions.drawing_utils

    # Load the pose landmarks from the JSON file
    with open(input_filename, 'r') as f:
        pose_landmarks_list = json.load(f)

    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (720, 720))

    # Loop over the pose landmarks and draw them on the image
    for i in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[i]

        # print(pose_landmarks))

        for j in range(len(pose_landmarks)):
            for k in range(len(pose_landmarks[j])):

                print(pose_landmarks[j]['x'])

                pose_landmarks[j]['x'] = pose_landmarks[j]['x'] * 720
                pose_landmarks[j]['y'] = pose_landmarks[j]['y'] * 720
                pose_landmarks[j]['z'] = pose_landmarks[j]['z'] * 720

            img = np.zeros((720, 720, 3), np.uint8)

            # Draw the pose annotation on the image
            mp_drawing.draw_landmarks(
                img,
                mp_pose.POSE_CONNECTIONS,
                [mp_pose.PoseLandmark(
                    landmark.x, landmark.y, landmark.z
                ) for landmark in pose_landmarks],
                mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(
                    color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            # Write the image to the output video
            out.write(img)

        # Release the VideoWriter object
        out.release()
    # Initialize mediapipe pose solution
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    import chardet

# Load the pose landmarks from the JSON file
    # Load the pose landmarks from the JSON file
    # Load the pose landmarks from the JSON file
    with open(input_filename, 'rb') as f:
        rawdata = f.read()
        result = chardet.detect(rawdata)
        encoding = result['encoding'] or 'utf-8'
        try:
            pose_landmarks_list = json.loads(rawdata.decode(encoding))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON data: {e}")
            pose_landmarks_list = []
    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (720, 720))

    # print(pose_landmarks_list)

    # Loop through the pose landmarks and create a new video frame for each set of landmarks
    for i in range(len(pose_landmarks_list)):

        # print(pose_landmarks)
        # create a new frame
        pose_landmarks = pose_landmarks_list[i]

        print(len(pose_landmarks))

        for j in range(len(pose_landmarks)):
            pose_landmarks[j]['x'] = pose_landmarks[j]['x'] * 720
            pose_landmarks[j]['y'] = pose_landmarks[j]['y'] * 720
            pose_landmarks[j]['z'] = pose_landmarks[j]['z'] * 720

        # print(pose_landmarks)

        img = np.zeros((720, 720, 3), np.uint8)

        # Draw the pose annotation on the image
        mp_pose.draw_landmarks(
            img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Write the video frame to the output video file
        out.write(img)

    # Release the VideoWriter object
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


# Call the function to extract pose from video and save it to a JSON file
input_folder = "Dataset/processedJSONDataset/"
output_folder = "Dataset/processVideoDataset/"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

transformVideo('landmarks.json','processedVideo.mp4')

# Load all files from the rawDataset folder and call transformVideo function
# for filename in os.listdir(input_folder):
#     input_file = os.path.join(input_folder, filename)
#     output_file = os.path.join(output_folder, filename.split(".")[0] + ".json")
#     print('Processing', input_file)
#     transformVideo(input_file, output_file)
#     print('Done processing', input_file)
