from project import PROJECT_DIR, to_snake_case, PAIRPLOTS_DIR
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
    pairplots_dir = os.path.join(PROJECT_DIR, PAIRPLOTS_DIR)
    if not os.path.exists(pairplots_dir):
        os.makedirs(pairplots_dir)

    for combination in combinations:
        # Plot pairwise relationships
        pairplot_file = to_snake_case(f'{combination[0]}__{combination[1]}.png')
        sns.pairplot(df[combination])
        plt.savefig(os.path.join(PROJECT_DIR, PAIRPLOTS_DIR, pairplot_file))
        plt.clf()

        print(f"Pairplot results for {combination[0]} and {combination[1]} saved to {pairplot_file}")