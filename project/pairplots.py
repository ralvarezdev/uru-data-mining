from project import to_snake_case, PAIRPLOTS_DIR
from project.combinations import generate_combinations
from project.load import load_transformed_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import os

if __name__ == '__main__':
    print("--- PAIRPLOTS ---\n")

    # Load the combinations of columns
    df = load_transformed_dataset()
    combinations = generate_combinations(2)

    # Verify that the pairplots directory exists, if not create it
    if not os.path.exists(PAIRPLOTS_DIR):
        os.makedirs(PAIRPLOTS_DIR)

    for combination in combinations:
        # Plot pairwise relationships
        pairplot_file = to_snake_case(f'{combination[0]}__{combination[1]}.png')
        sns.pairplot(df[combination])
        plt.savefig(os.path.join(PAIRPLOTS_DIR, pairplot_file))
        plt.clf()

        print(f"Pairplot results for {combination[0]} and {combination[1]} saved to {pairplot_file}")