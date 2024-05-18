from bvhtoolbox import BvhTree

# Path to a sample BVH file from the "AirplaneRiveting" dataset
sample_bvh_file = 'AirplaneRiveting/S01/PLNS01P01R01.bvh'

# Function to read and print information from a BVH file
def read_bvh_file(file_path):
    with open(file_path, 'r') as file:
        bvh_data = file.read()

    # Create a BvhTree object from the BVH data
    bvh_tree = BvhTree(bvh_data)

    # Print out some basic information about the BVH file
    print(f"Number of joints: {len(bvh_tree.get_joints())}")
    print(f"Frame time: {bvh_tree.frame_time}")
    print(f"Number of frames: {bvh_tree.nframes}")

# Call the function with the path to the sample BVH file
read_bvh_file(sample_bvh_file)
