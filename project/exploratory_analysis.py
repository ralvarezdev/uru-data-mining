import matplotlib.pyplot as plt
import seaborn as sns
import os
from project import PREVIEW_ROWS, COLUMNS_CHUNK_SIZE, to_snake_case, HISTOGRAMS_DIR, \
    BOXPLOTS_DIR, COUNTPLOTS_DIR
from project.load import load_dataset

if __name__ == '__main__':
    print("--- EXPLORATORY DATA ANALYSIS ---\n")

    # Load the dataset CSV file
    df = load_dataset()

    # Verify that the directories exist, if not create them
    for temp_dir in [HISTOGRAMS_DIR, BOXPLOTS_DIR, COUNTPLOTS_DIR]:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

    for column in df.columns:
        # Check if the column is numeric
        if df[column].dtype == 'float64' or df[column].dtype == 'int64':
            # Generate histogram for numeric columns
            sns.histplot(df[column])
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(HISTOGRAMS_DIR, f'{to_snake_case(column)}.png'))

            # Clear the plot
            plt.clf()

            print(f'Histogram of {column} saved to {os.path.join(HISTOGRAMS_DIR, f"{to_snake_case(column)}.png")}')

            # Check if it only contains '1' and '0' values
            if df[column].nunique() == 2 and set(df[column].unique()).issubset({0, 1}):
                continue

            # Generate boxplot for numeric columns
            sns.boxplot(x=df[column])
            plt.title(f'Boxplot of {column}')
            plt.xlabel(column)
            plt.savefig(os.path.join(BOXPLOTS_DIR, f'{to_snake_case(column)}.png'))

            # Clear the plot
            plt.clf()

            print(f'Boxplot of {column} saved to {os.path.join(BOXPLOTS_DIR, f"{to_snake_case(column)}.png")}')

        # Visualize the distribution of string values
        else:
            sns.countplot(x=column, data=df, order=df[column].value_counts().index)
            plt.title(f'Countplot of {column}')
            plt.ylabel('Frequency')
            plt.xlabel(column)
            plt.savefig(os.path.join(COUNTPLOTS_DIR, f'{to_snake_case(column)}.png'))

            # Clear the plot
            plt.clf()

            print(f'Countplot of {column} saved to {os.path.join(COUNTPLOTS_DIR, f"{to_snake_case(column)}.png")}')

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