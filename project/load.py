import os
import pandas as pd
from project import DATASET_DIR, ROAD_ACCIDENTS_DIR, ROAD_ACCIDENT_DATASET_FILE, TRANSFORMED_ROAD_ACCIDENT_DATASET_FILE

# Load the dataset
def load_dataset():
    # Get the current working directory
    current_dir = os.getcwd()

    # Get the path to the dataset
    dataset_path = os.path.join(current_dir, DATASET_DIR, ROAD_ACCIDENTS_DIR, ROAD_ACCIDENT_DATASET_FILE)

    # Load the dataset CSV file
    return pd.read_csv(dataset_path)

# Load the transformed dataset
def load_transformed_dataset():
    # Get the current working directory
    current_dir = os.getcwd()

    # Get the path to the dataset
    dataset_path = os.path.join(current_dir, DATASET_DIR, ROAD_ACCIDENTS_DIR, TRANSFORMED_ROAD_ACCIDENT_DATASET_FILE)

    # Load the dataset CSV file
    return pd.read_csv(dataset_path)