import ijson
import pandas as pd

# Function to read and convert the first few entries of a large JSON file into a pandas DataFrame
def inspect_json_file_to_df(file_path, num_entries=5):
    print(f"Opening file: {file_path}")  # Diagnostic print
    data = {'images': [], 'annotations': []}  # Initialize a dictionary to hold our data

    with open(file_path, 'rb') as file:
        print("File opened successfully.")  # Diagnostic print

        # Use ijson to iteratively parse the JSON file and add data to our dictionary
        images = ijson.items(file, 'images.item')
        for i, image in enumerate(images):
            if i >= num_entries:
                break
            data['images'].append(image)

        # Move back to the beginning of the file to parse annotations
        file.seek(0)
        annotations = ijson.items(file, 'annotations.item')
        for i, annotation in enumerate(annotations):
            if i >= num_entries:
                break
            data['annotations'].append(annotation)

    # Convert the lists in our dictionary to pandas DataFrames
    images_df = pd.DataFrame(data['images'])
    annotations_df = pd.DataFrame(data['annotations'])

    # Handle missing data by filling with appropriate values or dropping
    images_df.fillna(method='ffill', inplace=True)
    annotations_df.fillna(method='ffill', inplace=True)

    # Save the DataFrames to CSV files
    images_df.to_csv('images_dataframe.csv', index=False)
    annotations_df.to_csv('annotations_dataframe.csv', index=False)

    # Return the DataFrames
    return images_df, annotations_df

# Path to the DensePose-COCO train dataset
file_path = 'DensePose_COCO/densepose_coco_2014_train.json'

# Call the function to inspect the dataset and convert to DataFrames
images_df, annotations_df = inspect_json_file_to_df(file_path)

# Print the first few rows of each DataFrame to verify
print(images_df.head())
print(annotations_df.head())
