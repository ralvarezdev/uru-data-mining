import pandas as pd
from project import ROAD_ACCIDENT_DATASET_FILE, TRANSFORMED_ROAD_ACCIDENT_DATASET_FILE

# Load the dataset
def load_dataset():
    return pd.read_csv(ROAD_ACCIDENT_DATASET_FILE)

# Load the transformed dataset
def load_transformed_dataset():
    return pd.read_csv(TRANSFORMED_ROAD_ACCIDENT_DATASET_FILE)