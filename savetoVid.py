import os
import cv2
import mediapipe as mp
import numpy as np
import imageio

# Create a function to extract pose from video and save it to a video input file


def transformVideo(input_filename, output_filename):
    # Initialize mediapipe pose solution
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    # Take video input for pose detection
    cap = cv2.VideoCapture(input_filename)

    # Prepare the writer to save frames as AVI video
    writer = imageio.get_writer(output_filename, fps=cap.get(cv2.CAP_PROP_FPS))

    # Read each frame/image from the capture object
    while True:
        ret, img = cap.read()
        if not ret:
            break

        # Resize the image/frame to fit the screen
        img = cv2.resize(img, (720, 720))

        # Do pose detection
        results = pose.process(img)

        # Draw the detected pose on the original video/live stream
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                               mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                               )

        # Display the frame with the extracted pose
        # cv2.imshow("Extracted Pose", img)

        # Display the frame with the detected pose
        h, w, c = img.shape   # get shape of original frame
        # create blank image with original frame size
        opImg = np.zeros([h, w, c])
        # set white background. put 0 if you want to make it black
        opImg.fill(0)

        # draw extracted pose on black white image
        mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(
                                   color=(0, 255, 0), thickness=4, circle_radius=4),
                               mp_draw.DrawingSpec(
                                   color=(0, 0, 255), thickness=4, circle_radius=6)
                               )

        # Convert the image to 8-bit depth
        opImg = cv2.normalize(opImg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Append the converted image to the video writer
        writer.append_data(cv2.cvtColor(opImg, cv2.COLOR_BGR2RGB))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer
    cap.release()
    writer.close()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


# Call the function to extract pose from video and save it to a video input file
input_folder = "Dataset/rawDataset/"
output_folder = "Dataset/processedDataset/"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load all files from the rawDataset folder and call transformVideo function
for filename in os.listdir(input_folder):
    input_file = os.path.join(input_folder, filename)
    output_file = os.path.join(output_folder, filename.split(".")[0] + ".avi")
    print('Processing', input_file)
    transformVideo(input_file, output_file)
    print('Done processing', input_file)
