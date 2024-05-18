import math

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between three points.
    Points are given as tuples of (x, y) coordinates.
    """
    # Get the coordinates of the points
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Calculate the vectors from point 2 to point 1 and point 3
    vector21 = (x1 - x2, y1 - y2)
    vector23 = (x3 - x2, y3 - y2)

    # Calculate the dot product and magnitude of the vectors
    dot_product = vector21[0] * vector23[0] + vector21[1] * vector23[1]
    magnitude21 = math.sqrt(vector21[0]**2 + vector21[1]**2)
    magnitude23 = math.sqrt(vector23[0]**2 + vector23[1]**2)

    # Calculate the angle in radians and then convert to degrees
    angle = math.acos(dot_product / (magnitude21 * magnitude23))
    angle_degrees = math.degrees(angle)

    return angle_degrees

def calculate_ergonomic_risk(pose_landmarks):
    """
    Calculate ergonomic risk based on pose landmarks.
    This function will analyze the pose landmarks and calculate an ergonomic risk score
    based on criteria similar to the RULA method.
    """
    ergonomic_risk_score = 0
    # Define risky angle thresholds for different joints
    ELBOW_RISKY_ANGLE = 150  # Example threshold for the elbow joint
    ELBOW_RISK_POINTS = 1    # Example risk points for the elbow joint

    # Define RULA scoring system for upper arm, lower arm, and wrist
    UPPER_ARM_SCORES = {90: 3, 60: 2, 20: 1}  # Example scores based on degrees
    LOWER_ARM_SCORES = {60: 2, 20: 1}
    WRIST_SCORES = {15: 3, 0: 1}

    # Define RULA scoring system for neck, trunk, and legs
    NECK_SCORES = {20: 3, 10: 2, 0: 1}
    TRUNK_SCORES = {20: 3, 10: 2, 0: 1}
    LEG_SCORES = {2: 2, 1: 1}

    # Example of calculating risk score for one joint (e.g., elbow)
    # Assume we have the coordinates for the left shoulder, elbow, and wrist
    left_shoulder = pose_landmarks['left_shoulder']
    left_elbow = pose_landmarks['left_elbow']
    left_wrist = pose_landmarks['left_wrist']

    # Calculate the angle of the elbow joint
    elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    # Check if the elbow angle exceeds the risky threshold
    if elbow_angle > ELBOW_RISKY_ANGLE:
        ergonomic_risk_score += ELBOW_RISK_POINTS

    # Replace placeholder points with actual pose landmark data
    # The following variables should be assigned the coordinates of the respective body parts
    # These coordinates would typically come from the pose estimation model's output
    left_hand = pose_landmarks['left_hand_index']
    left_fingers = pose_landmarks['left_fingers']
    head = pose_landmarks['head']
    neck = pose_landmarks['neck']
    upper_back = pose_landmarks['upper_back']
    lower_back = pose_landmarks['lower_back']
    hips = pose_landmarks['hips']
    knees = pose_landmarks['knees']
    ankles = pose_landmarks['ankles']

    # Calculate the angle for the upper arm and determine the score
    upper_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    upper_arm_score = UPPER_ARM_SCORES[min([key for key in UPPER_ARM_SCORES if upper_arm_angle >= key])]

    # Calculate the angle for the lower arm and determine the score
    lower_arm_angle = calculate_angle(left_elbow, left_wrist, left_hand)
    lower_arm_score = LOWER_ARM_SCORES[min([key for key in LOWER_ARM_SCORES if lower_arm_angle >= key])]

    # Calculate the angle for the wrist and determine the score
    wrist_angle = calculate_angle(left_wrist, left_hand, left_fingers)
    wrist_score = WRIST_SCORES[min([key for key in WRIST_SCORES if wrist_angle >= key])]

    # Calculate the angle for the neck and determine the score
    neck_angle = calculate_angle(head, neck, upper_back)
    neck_score = NECK_SCORES[min([key for key in NECK_SCORES if neck_angle >= key])]

    # Calculate the angle for the trunk and determine the score
    trunk_angle = calculate_angle(upper_back, lower_back, hips)
    trunk_score = TRUNK_SCORES[min([key for key in TRUNK_SCORES if trunk_angle >= key])]

    # Calculate the angle for the legs and determine the score
    leg_angle = calculate_angle(hips, knees, ankles)
    leg_score = LEG_SCORES[min([key for key in LEG_SCORES if leg_angle >= key])]

    # Sum the scores for each body part to get an overall ergonomic risk score
    overall_ergonomic_risk_score = sum([upper_arm_score, lower_arm_score, wrist_score, neck_score, trunk_score, leg_score])

    # Determine the action level based on the overall ergonomic risk score
    if overall_ergonomic_risk_score >= 7:
        action_level = 'Investigate and implement change'
    elif overall_ergonomic_risk_score >= 5:
        action_level = 'Further investigation and change soon'
    else:
        action_level = 'Acceptable'

    return overall_ergonomic_risk_score, action_level

import cv2
import mediapipe as mp

def main(input_video_path):
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Read the input video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose estimation
        results = pose.process(frame_rgb)

        # Check if any landmarks are detected
        if results.pose_landmarks:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark

            # Prepare the pose landmarks dictionary
            pose_landmarks = {
                'left_shoulder': (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),
                'right_shoulder': (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
                'left_elbow': (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y),
                'right_elbow': (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y),
                'left_wrist': (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y),
                'right_wrist': (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y),
                'left_hand_index': (landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y),
                'right_hand_index': (landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y),
                'left_hip': (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y),
                'right_hip': (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y),
                'left_knee': (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y),
                'right_knee': (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y),
                'left_ankle': (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y),
                'right_ankle': (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y),
                # Add other landmarks if needed
            }
            # Ensure all required landmarks are present before calculating risk
            required_landmarks = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                                  'left_wrist', 'right_wrist', 'left_hand_index', 'right_hand_index',
                                  'left_hip', 'right_hip', 'left_knee', 'right_knee',
                                  'left_ankle', 'right_ankle']
            if all(key in pose_landmarks for key in required_landmarks):
                # Calculate ergonomic risk score
                ergonomic_risk_score, action_level = calculate_ergonomic_risk(pose_landmarks)
                print(f"Ergonomic Risk Score: {ergonomic_risk_score}, Action Level: {action_level}")
            else:
                print("Error: Not all required landmarks were detected.")

        # Display the frame with pose estimation (optional)
        # cv2.imshow('Pose Estimation', frame)
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break

    cap.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    # Placeholder for the input video path
    input_video_path = 'path_to_input_video.mp4'
    main(input_video_path)
