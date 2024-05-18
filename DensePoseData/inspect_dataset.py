import ijson

# Function to read and print the first few entries of a large JSON file
def inspect_json_file(file_path, num_entries=5):
    print(f"Opening file: {file_path}")  # Diagnostic print
    with open(file_path, 'rb') as file:
        print("File opened successfully.")  # Diagnostic print
        # Use ijson to iteratively parse the JSON file
        # Print the details of the first few 'images' entries
        images = ijson.items(file, 'images.item')
        for i, image in enumerate(images):
            if i >= num_entries:
                break
            print(f"Image {i}: {image}")  # Diagnostic print

        # Move back to the beginning of the file to parse annotations
        file.seek(0)
        # Print the details of the first few 'annotations' entries
        annotations = ijson.items(file, 'annotations.item')
        for i, annotation in enumerate(annotations):
            if i >= num_entries:
                break
            print(f"Annotation {i}: {annotation}")  # Diagnostic print

# Path to the DensePose-COCO train dataset
file_path = 'DensePose_COCO/densepose_coco_2014_train.json'

# Call the function to inspect the dataset
inspect_json_file(file_path)
