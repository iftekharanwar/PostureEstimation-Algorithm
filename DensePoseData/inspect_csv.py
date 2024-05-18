import pandas as pd

# Load a small sample of the CSV file to understand its structure
def load_csv_sample(file_path, sample_size=5):
    data = pd.read_csv(file_path, nrows=sample_size)
    print(data.head())

# Path to the CSV file
csv_file_path = 'AirplaneRiveting/CSV/PLNS01P01R01_pos.csv'

# Call the function to load and print the sample data
load_csv_sample(csv_file_path)
