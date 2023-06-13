def transformVideo(input_filename, output_filename):
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
