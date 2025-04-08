from project import HEATMAPS_DIR, to_snake_case
from project.combinations import generate_combinations
from project.load import load_transformed_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    print("--- HEATMAPS ---\n")

    # Load the combinations of columns
    df = load_transformed_dataset()
    combinations = generate_combinations(2)

    # Verify that the heatmaps directory exists, if not create it
    if not os.path.exists(HEATMAPS_DIR):
        os.makedirs(HEATMAPS_DIR)

    for combination in combinations:
        # Get the columns for the heatmap
        columns = list(combination)

        # Select the relevant columns from the DataFrame
        columns_df = df[columns]

        # Compute the correlation matrix
        correlation_matrix = columns_df.corr()

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

        # Save the heatmap
        heatmap_file = os.path.join(HEATMAPS_DIR, to_snake_case(f'{combination[0]}__{combination[1]}.png'))
        plt.savefig(heatmap_file)
        plt.clf()

        print(f"Heatmap results for {combination[0]} and {combination[1]} saved to {heatmap_file}")