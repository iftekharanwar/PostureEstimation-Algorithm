from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.viz_tools import *
import os

def extract_pose_data_from_bvh(bvh_file_path):
    try:
        # Parse the BVH file
        parser = BVHParser()
        parsed_data = parser.parse(bvh_file_path)

        # Preprocess the data
        # This can include filtering, down-sampling, etc.
        # For now, we'll just convert it to a pandas DataFrame for easy manipulation
        pp = MocapParameterizer('position')
        processed_data = pp.fit_transform([parsed_data])

        # Visualize the pose data (optional, for debugging/verification)
        # This will create a simple animation of the motion capture data
        skel = parsed_data.skeleton
        anim = parsed_data.values
        render_mp4_from_bvh(skel, anim, 'output_animation.mp4')

        return processed_data
    except Exception as e:
        print(f"An error occurred while processing the BVH file: {e}")
        return None

# Example usage:
# Replace 'path_to_bvh_file.bvh' with the actual path to a BVH file
# Assuming the AirplaneRiveting dataset is extracted in the current directory
bvh_files = [file for file in os.listdir('.') if file.endswith('.bvh')]
for bvh_file in bvh_files:
    pose_data = extract_pose_data_from_bvh(bvh_file)
    if pose_data is not None:
        print(f"Pose data extracted for {bvh_file}")
    else:
        print(f"Failed to extract pose data for {bvh_file}")
