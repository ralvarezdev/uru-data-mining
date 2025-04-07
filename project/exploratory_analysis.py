import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

from project import PROJECT_DIR, CHARTS_DIR, PREVIEW_ROWS, COLUMNS_CHUNK_SIZE
from project.load import load_dataset

# Convert camel case to snake case
def to_snake_case(s):
    # Drop '\' and '/' characters
    s = re.sub(r'[\\/]', '', s)

    # Replace spaces with underscores
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()
    s = re.sub(r'\s+', '', s)
    return s

if __name__ == '__main__':
    print("--- EXPLORATORY DATA ANALYSIS ---\n")

    # Load the dataset CSV file
    df = load_dataset()

    # Verify that the charts directory exists, if not create it
    charts_dir = os.path.join(PROJECT_DIR, CHARTS_DIR)
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)

    for column in df.columns:
        # Check if the column is numeric
        if df[column].dtype == 'float64' or df[column].dtype == 'int64':
            # Generate histogram for numeric columns
            sns.histplot(df[column])
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(PROJECT_DIR, CHARTS_DIR, f'{to_snake_case(column)}_histogram.png'))

            # Clear the plot
            plt.clf()

            # Check if it only contains '1' and '0' values
            if df[column].nunique() == 2 and set(df[column].unique()).issubset({0, 1}):
                continue

            # Generate boxplot for numeric columns
            sns.boxplot(x=df[column])
            plt.title(f'Boxplot of {column}')
            plt.xlabel(column)
            plt.savefig(os.path.join(PROJECT_DIR, CHARTS_DIR, f'{to_snake_case(column)}_boxplot.png'))

            # Clear the plot
            plt.clf()

        # Visualize the distribution of string values
        else:
            sns.countplot(x=column, data=df, order=df[column].value_counts().index)
            plt.title(f'Histogram of {column}')
            plt.ylabel('Frequency')
            plt.xlabel(column)
            plt.savefig(os.path.join(PROJECT_DIR, CHARTS_DIR, f'{to_snake_case(column)}_histogram.png'))
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