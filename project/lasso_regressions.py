from project import LASSO_REGRESSIONS_DIR, LASSO_REGRESSIONS_FILE
from project.combinations import generate_combinations
from project.load import load_transformed_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
import os

# Constants
MAX_SIZE = 3

if __name__ == '__main__':
    print("--- LASSO REGRESSIONS ---\n")

    # Load the combinations of columns
    df = load_transformed_dataset()
    combinationsList=[]
    for i in range(2, MAX_SIZE + 1):
        combinationsList.append(generate_combinations(i))
        print(f"Generated {len(combinationsList[-1])} combinations of size {i}")

    # Verify that the Lasso Regressions directory exists, if not create it
    if not os.path.exists(LASSO_REGRESSIONS_DIR):
        os.makedirs(LASSO_REGRESSIONS_DIR)

    # Iterate through the combinations
    r2_combinations = {}
    for combinations in combinationsList:
        for combination in combinations:
            x = list(combination[:-1])
            y = combination[-1]

            X = df[x]
            Y = df[y]

            # Split the dataset into training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            # Use LassoCV to find the optimal alpha
            lasso_cv = LassoCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5, random_state=42)
            lasso_cv.fit(X_train, Y_train)

            # Optimal alpha
            optimal_alpha = lasso_cv.alpha_

            # Evaluate the model
            r2 = lasso_cv.score(X_test, Y_test)

            # Save the combination if R^2
            r2_combinations[combination] = [r2, optimal_alpha]

    # Open the file in write mode
    with open(os.path.join(LASSO_REGRESSIONS_FILE, LASSO_REGRESSIONS_FILE), 'w') as file:
        file.write(str(r2_combinations))

    print(f"Lasso Regressions results saved to {LASSO_REGRESSIONS_FILE}")