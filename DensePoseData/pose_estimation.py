import math
import cv2
import mediapipe as mp
import csv
import os
import argparse
from collections import namedtuple

mp_pose = mp.solutions.pose

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between three points.
    Points are given as tuples of (x, y) coordinates.
    If any point is None, return 0 as the default angle.
    """
    # Check if any of the points are None and return 0 if so
    if p1 is None or p2 is None or p3 is None:
        return 0

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

def is_ulnar_deviation(wrist_landmarks, pinky_landmarks, elbow_landmarks):
    # Calculate the angle between the wrist, pinky finger, and elbow
    ulnar_deviation_angle = calculate_angle(wrist_landmarks, pinky_landmarks, elbow_landmarks)
    # Check if the ulnar deviation angle exceeds the threshold for ulnar deviation
    return ulnar_deviation_angle > 20  # Threshold angle for ulnar deviation

def is_foot_supported(foot_landmark, hip_landmark):
    # Check if the foot is below the hip, which would indicate it is supported
    # This is a simplified assumption that the foot is supported if it is below the hip
    # In a real-world scenario, additional checks would be needed for accuracy
    # foot_landmark and hip_landmark are tuples of (x, y) coordinates
    return foot_landmark[1] > hip_landmark[1]

def is_shoulder_elevated(shoulder_landmarks, reference_landmarks):
    """
    Check if the shoulder is elevated.
    This function compares the y-coordinate of the shoulder to a reference point (e.g., hip).
    An elevated shoulder is typically higher (smaller y-value) than the reference.
    """
    shoulder_y = shoulder_landmarks[1]
    reference_y = reference_landmarks[1]
    return shoulder_y < reference_y

def is_arm_abducted(shoulder_landmarks, elbow_landmarks):
    """
    Check if the arm is abducted.
    This function calculates the horizontal distance between the shoulder and elbow.
    A significant horizontal distance indicates abduction.
    """
    shoulder_x = shoulder_landmarks[0]
    elbow_x = elbow_landmarks[0]
    return abs(elbow_x - shoulder_x) > 0.1  # Threshold for abduction to be determined

def is_leaning_or_supporting(shoulder_landmarks, elbow_landmarks):
    """
    Check if the person is leaning or if the arm is supported.
    This function checks the vertical alignment of the shoulder and elbow.
    Leaning or support is indicated by the elbow being vertically aligned or below the shoulder.
    """
    shoulder_y = shoulder_landmarks[1]
    elbow_y = elbow_landmarks[1]
    return elbow_y >= shoulder_y

def extract_landmarks_from_csv_row(row):
    """
    Extract landmarks from a CSV row and create a pose_landmarks object.
    The CSV columns are named with the pattern 'JointName.x', 'JointName.y', and 'JointName.z'.
    """
    class Landmarks:
        def __init__(self):
            self.landmark = {}

    pose_landmarks = Landmarks()

    # Mapping of CSV column names to MediaPipe PoseLandmark enum names
    landmark_mapping = {
        'RightShoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
        'RightElbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
        'RightWrist': mp_pose.PoseLandmark.RIGHT_WRIST,
        'RightPinky': mp_pose.PoseLandmark.RIGHT_PINKY,
        'LeftShoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
        'LeftElbow': mp_pose.PoseLandmark.LEFT_ELBOW,
        'LeftWrist': mp_pose.PoseLandmark.LEFT_WRIST,
        'LeftPinky': mp_pose.PoseLandmark.LEFT_PINKY,
        'RightHip': mp_pose.PoseLandmark.RIGHT_HIP,
        'LeftHip': mp_pose.PoseLandmark.LEFT_HIP,
        'RightKnee': mp_pose.PoseLandmark.RIGHT_KNEE,
        'LeftKnee': mp_pose.PoseLandmark.LEFT_KNEE,
        'RightAnkle': mp_pose.PoseLandmark.RIGHT_ANKLE,
        'LeftAnkle': mp_pose.PoseLandmark.LEFT_ANKLE,
        'RightFootIndex': mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
        'LeftFootIndex': mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
        'Nose': mp_pose.PoseLandmark.NOSE,
        'RightPinky': mp_pose.PoseLandmark.RIGHT_PINKY,
        'LeftShoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
        'LeftElbow': mp_pose.PoseLandmark.LEFT_ELBOW,
        'LeftWrist': mp_pose.PoseLandmark.LEFT_WRIST,
        'LeftPinky': mp_pose.PoseLandmark.LEFT_PINKY,
        'RightHip': mp_pose.PoseLandmark.RIGHT_HIP,
        'LeftHip': mp_pose.PoseLandmark.LEFT_HIP,
        'RightKnee': mp_pose.PoseLandmark.RIGHT_KNEE,
        'LeftKnee': mp_pose.PoseLandmark.LEFT_KNEE,
        'RightAnkle': mp_pose.PoseLandmark.RIGHT_ANKLE,
        'LeftAnkle': mp_pose.PoseLandmark.LEFT_ANKLE,
        'RightFootIndex': mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
        'LeftFootIndex': mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
        'Nose': mp_pose.PoseLandmark.NOSE,
        # ... other mappings as needed
    }

    # Extract landmarks using the mapping
    for landmark_name, pose_landmark in landmark_mapping.items():
        x_column = f'{landmark_name}.x'
        y_column = f'{landmark_name}.y'
        z_column = f'{landmark_name}.z'
        if x_column in row and y_column in row and z_column in row:
            # Create a namedtuple to mimic the MediaPipe landmark structure
            LandmarkPoint = namedtuple('LandmarkPoint', ['x', 'y', 'z'])
            # Convert the CSV string values to float and create the LandmarkPoint
            pose_landmarks.landmark[pose_landmark] = LandmarkPoint(
                x=float(row[x_column]),
                y=float(row[y_column]),
                z=float(row[z_column])
            )
        else:
            # If any of the required columns are missing, add a None value for the landmark
            pose_landmarks.landmark[pose_landmark] = None

    return pose_landmarks

def calculate_ergonomic_risk(pose_landmarks):
    """
    Calculate ergonomic risk based on pose landmarks.
    This function will analyze the pose landmarks and calculate an ergonomic risk score
    based on criteria similar to the RULA method.
    """
    # Initialize the ergonomic risk score
    ergonomic_risk_score = 0
    print("Initial ergonomic risk score:", ergonomic_risk_score)  # Diagnostic print

    # Helper function to get a landmark if it exists, otherwise return None
    def get_landmark(landmark):
        return pose_landmarks.landmark.get(landmark)

    # Right side
    right_shoulder = get_landmark(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    right_elbow = get_landmark(mp_pose.PoseLandmark.RIGHT_ELBOW)
    right_wrist = get_landmark(mp_pose.PoseLandmark.RIGHT_WRIST)
    right_pinky = get_landmark(mp_pose.PoseLandmark.RIGHT_PINKY)

    right_elbow_angle = 0  # Initialize to default value
    # Check if required landmarks are available before calculating angles
    if right_shoulder and right_elbow and right_wrist:
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        print(f"Calculated right elbow angle: {right_elbow_angle}")  # Diagnostic print

        # Assign scores based on RULA criteria for the right upper arm
        if 20 <= right_elbow_angle <= 45:
            ergonomic_risk_score += 2
            print("Right elbow angle between 20 and 45 degrees, adding 2 to risk score")  # Diagnostic print
        elif 45 < right_elbow_angle <= 90:
            ergonomic_risk_score += 3
            print("Right elbow angle between 45 and 90 degrees, adding 3 to risk score")  # Diagnostic print
        elif right_elbow_angle > 90:
            ergonomic_risk_score += 4
            print("Right elbow angle greater than 90 degrees, adding 4 to risk score")  # Diagnostic print
        else:
            ergonomic_risk_score += 1
            print("Right elbow angle less than 20 degrees, adding 1 to risk score")  # Diagnostic print
        print(f"Right upper arm score after evaluation: {ergonomic_risk_score}")  # Diagnostic print

        # Adjust score for wrist posture based on RULA criteria
        if right_wrist and right_pinky and is_ulnar_deviation(right_wrist, right_pinky, right_elbow):
            ergonomic_risk_score += 1  # Add score for ulnar deviation
            print("Ulnar deviation detected, adding 1 to risk score")  # Diagnostic print
        print(f"Right wrist score after evaluation: {ergonomic_risk_score}")  # Diagnostic print

    # Adjust score for lower arm posture based on RULA criteria
    lower_arm_angle_threshold = 60  # Threshold angle for lower arm posture
    if right_elbow_angle and (right_elbow_angle < lower_arm_angle_threshold or right_elbow_angle > (180 - lower_arm_angle_threshold)):
        ergonomic_risk_score += 1  # Add score if the lower arm angle is too acute or too obtuse
        print(f"Lower arm posture adjustment: {ergonomic_risk_score}")  # Diagnostic print

    # Calculate neck angle for flexion/extension
    left_shoulder = get_landmark(mp_pose.PoseLandmark.LEFT_SHOULDER)
    right_shoulder = get_landmark(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    nose = get_landmark(mp_pose.PoseLandmark.NOSE)

    if left_shoulder and right_shoulder and nose:
        neck = ((left_shoulder.x + right_shoulder.x) / 2,
                (left_shoulder.y + right_shoulder.y) / 2)
        head = (nose.x, nose.y)
        upper_back = (left_shoulder.x, left_shoulder.y)
        neck_angle = calculate_angle(upper_back, neck, head)

        # Neck scoring based on flexion/extension
        if neck_angle < 10:
            ergonomic_risk_score += 1
        elif 10 <= neck_angle < 20:
            ergonomic_risk_score += 2
        else:
            ergonomic_risk_score += 3
        print(f"Neck angle evaluation: {ergonomic_risk_score}")  # Diagnostic print

    # Calculate trunk flexion/extension angle if landmarks are available
    left_shoulder = get_landmark(mp_pose.PoseLandmark.LEFT_SHOULDER)
    right_hip = get_landmark(mp_pose.PoseLandmark.RIGHT_HIP)
    left_hip = get_landmark(mp_pose.PoseLandmark.LEFT_HIP)
    if left_shoulder and right_hip and left_hip:
        pelvis = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)  # Midpoint of hips as pelvis
        trunk_flexion_extension_angle = calculate_angle(left_shoulder, (left_shoulder.x, pelvis.y), pelvis)
        # Trunk scoring based on flexion/extension and twist
        if trunk_flexion_extension_angle < 10:
            ergonomic_risk_score += 1
        elif 10 <= trunk_flexion_extension_angle < 20:
            ergonomic_risk_score += 2
        else:
            ergonomic_risk_score += 3

    # Calculate leg support score based on the position of the feet relative to the hips
    right_foot = get_landmark(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
    left_foot = get_landmark(mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
    if right_foot and left_foot and right_hip and left_hip:
        right_foot_supported = is_foot_supported(right_foot, right_hip)
        left_foot_supported = is_foot_supported(left_foot, left_hip)
        leg_support_score = 1 if right_foot_supported and left_foot_supported else 2
        ergonomic_risk_score += leg_support_score
        print(f"Leg support evaluation: {ergonomic_risk_score}")  # Diagnostic print

    # Adjust score for shoulder elevation, arm abduction, leaning, or arm support
    if right_shoulder and right_hip:
        reference_landmarks = (right_hip.x, right_hip.y)
        if is_shoulder_elevated(right_shoulder, reference_landmarks):
            ergonomic_risk_score += 1
        if right_elbow and is_arm_abducted(right_shoulder, right_elbow):
            ergonomic_risk_score += 1
        if right_elbow and is_leaning_or_supporting(right_shoulder, right_elbow):
            ergonomic_risk_score += 1  # Adjusted to add score if leaning or arm is supported

    # Continue with the rest of the ergonomic risk calculations...
    # (The rest of the function remains unchanged)

    # Left side
    left_elbow = get_landmark(mp_pose.PoseLandmark.LEFT_ELBOW)
    left_wrist = get_landmark(mp_pose.PoseLandmark.LEFT_WRIST)
    left_pinky = get_landmark(mp_pose.PoseLandmark.LEFT_PINKY)
    left_elbow_angle = 0  # Initialize to default value
    if left_shoulder and left_elbow and left_wrist:
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

        # Assign scores based on RULA criteria for the left upper arm
        if 20 <= left_elbow_angle <= 45:
            ergonomic_risk_score += 2
        elif 45 < left_elbow_angle <= 90:
            ergonomic_risk_score += 3
        elif left_elbow_angle > 90:
            ergonomic_risk_score += 4
        else:
            ergonomic_risk_score += 1

        # Adjust score for wrist posture based on RULA criteria
        if left_wrist and left_pinky and is_ulnar_deviation(left_wrist, left_pinky, left_elbow):
            ergonomic_risk_score += 1  # Add score for ulnar deviation

    # Adjust score for shoulder elevation, arm abduction, leaning, or arm support
    if left_shoulder and left_hip:
        reference_landmarks = (left_hip.x, left_hip.y)
        if is_shoulder_elevated(left_shoulder, reference_landmarks):
            ergonomic_risk_score += 1
        if left_elbow and is_arm_abducted(left_shoulder, left_elbow):
            ergonomic_risk_score += 1
        if left_elbow and is_leaning_or_supporting(left_shoulder, left_elbow):
            ergonomic_risk_score += 1  # Adjusted to add score if leaning or arm is supported

    # Determine action level based on ergonomic risk score
    if ergonomic_risk_score >= 7:
        action_level = "Immediate action required"
    elif ergonomic_risk_score >= 5:
        action_level = "Further investigation and change soon"
    elif ergonomic_risk_score >= 3:
        action_level = "Monitor and review"
    else:
        action_level = "Low risk - maintain current practices"

    return ergonomic_risk_score, action_level

def main(csv_file_path):
    # Initialize CSV file for storing results
    with open('ergonomic_risk_scores.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Ergonomic Risk Score', 'Action Level'])

        # Load the CSV file containing pose data
        with open(csv_file_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            frame_number = 0
            print("Starting CSV data processing loop...")  # Confirm entry into the loop

            for row in csv_reader:
                # Extract landmarks from the CSV row
                pose_landmarks = extract_landmarks_from_csv_row(row)

                # Calculate the ergonomic risk score using the extracted landmarks
                ergonomic_risk_score, action_level = calculate_ergonomic_risk(pose_landmarks)

                # Print the frame number, ergonomic risk score, and action level to the console
                print(f"Frame: {frame_number}, Ergonomic Risk Score: {ergonomic_risk_score}, Action Level: {action_level}")

                # Write the actual risk score and action level to the CSV
                writer.writerow([frame_number, ergonomic_risk_score, action_level])
                frame_number += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a CSV file and calculate ergonomic risk scores.')
    parser.add_argument('csv_file_path', help='Path to the input CSV file containing pose data')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Expand the user's home directory if the tilde is used in the CSV file path
    csv_file_path = os.path.expanduser(args.csv_file_path)

    # Call the main function with the provided CSV file path
    main(csv_file_path)
