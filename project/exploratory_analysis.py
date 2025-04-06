import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Constants
DATASET_DIR = 'dataset'
ROAD_ACCIDENTS_DIR = 'road-accidents'
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = 'charts'
ROAD_ACCIDENT_DATASET_FILE = 'road_accident_dataset.csv'
PREVIEW_ROWS = 10
COLUMNS_CHUNK_SIZE = 5

# Convert camel case to snake case
def to_snake_case(s):
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()
    s = re.sub(r'\s+', '_', s)
    return s

if __name__ == '__main__':
    print("---  EXPLORATORY DATA ANALYSIS ---\n")

    # Get the current working directory
    current_dir = os.getcwd()

    # Get the path to the dataset
    dataset_path = os.path.join(current_dir, DATASET_DIR, ROAD_ACCIDENTS_DIR, ROAD_ACCIDENT_DATASET_FILE)

    # Load the dataset CSV file
    df = pd.read_csv(dataset_path)

    # Verify that the charts directory exists, if not create it
    charts_dir = os.path.join(PROJECT_DIR, CHARTS_DIR)
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)

    # Generate the histogram chart for the columns
    for column in df.columns:
        if df[column].dtype == 'float64' or df[column].dtype == 'int64':
            # Generate histogram for numeric columns
            sns.histplot(df[column])
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(PROJECT_DIR, CHARTS_DIR, f'{to_snake_case(column)}_histogram.png'))

            # Clear the plot
            plt.clf()

    # Split by columns chunk size
    chunks = []
    chunks_str=[]
    columns = df.columns
    for i in range(0, len(columns), COLUMNS_CHUNK_SIZE):
        chunk = columns[i:i + COLUMNS_CHUNK_SIZE]
        chunks.append(chunk)
        chunks_str.append(', '.join(chunk))

    # Preview the data
    for i in range(len(chunks)):
        print(f"\n--- DATA PREVIEW ({chunks_str[i]}) ---\n")
        print(df[chunks[i]].head(n=PREVIEW_ROWS))

    # Display the types of each column
    print("\n--- DATA TYPES ---\n")
    print(df.info())

    # Display the statics of the dataset
    for i in range(len(chunks)):
        print(f"\n--- DATA STATISTICS ({chunks_str[i]}) ---\n")
        print(df[chunks[i]].describe())

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Fill NaN values with the median of numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # Fill NaN values with the most frequent value of non-numeric columns
    non_numeric_columns = df.select_dtypes(exclude=['float64', 'int64']).columns
    df[non_numeric_columns] = df[non_numeric_columns].apply(lambda x: x.fillna(x.mode()[0]))