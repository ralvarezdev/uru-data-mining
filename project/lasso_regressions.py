from project import PROJECT_DIR, LASSO_REGRESSIONS_DIR, LASSO_REGRESSIONS_FILE
from project.combinations import generate_combinations
from project.load import load_transformed_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
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

    # Verify that the Lasso Regression directory exists, if not create it
    lasso_regressions_dir = os.path.join(PROJECT_DIR, LASSO_REGRESSIONS_DIR)
    if not os.path.exists(lasso_regressions_dir):
        os.makedirs(lasso_regressions_dir)

    # Iterate through the combinations
    high_r2_combinations = {}
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
            # print(f"Optimal alpha: {optimal_alpha}")

            # Evaluate the model
            r2 = lasso_cv.score(X_test, Y_test)
            # print(f"R^2 Score: {r2}")

            # Ignore the R^2 value if it is too low
            if abs(r2) <= 0.001:
                continue

            # Save the combination if R^2 is high
            high_r2_combinations[combination] = [r2, optimal_alpha]

    # Check if there are any high R^2 combinations
    if not high_r2_combinations:
        print("No high R^2 combinations found")

    else:
        # Open the file in write mode
        with open(os.path.join(PROJECT_DIR, LASSO_REGRESSIONS_FILE, LASSO_REGRESSIONS_FILE), 'w') as file:
            file.write(str(high_r2_combinations))

        print(f"Lasso Regressions results saved to {LASSO_REGRESSIONS_FILE}")