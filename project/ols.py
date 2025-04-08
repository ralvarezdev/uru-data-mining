from project import PROJECT_DIR, OLS_DIR, to_snake_case
from project.combinations import generate_combinations
from project.load import load_transformed_dataset
import statsmodels.api as sm
import os

# Constants
MAX_SIZE = 10

if __name__ == '__main__':
    print("--- OLS ---\n")

    # Load the combinations of columns
    df = load_transformed_dataset()
    combinationsList=[]
    for i in range(2, MAX_SIZE + 1):
        combinationsList.append(generate_combinations(i))
        print(f"Generated {len(combinationsList[-1])} combinations of size {i}")

    # Verify that the OLS directory exists, if not create it
    ols_dir = os.path.join(PROJECT_DIR, OLS_DIR)
    if not os.path.exists(ols_dir):
        os.makedirs(ols_dir)

    for combinations in combinationsList:
        for combination in combinations:
            if len(combination) == 2:
                x = combination[0]
            else:
                x = list(combination[:-1])
            y = combination[-1]

            X = df[x]
            Y = df[y]

            # Add a constant to the model (intercept)
            X = sm.add_constant(X)

            # Fit the regression model for the given combination
            model = sm.OLS(Y, X).fit()
            print(f"OLS results for {x} and {y}: {model.rsquared}")

            # Ignore the R^2 value if it is too low
            if model.rsquared <= 0.001:
                continue

            # Open the file in write mode
            ols_file= to_snake_case(f'{"__".join(combination)}.txt')
            with open(os.path.join(PROJECT_DIR, OLS_DIR, ols_file), 'w') as file:
                file.write(str(model.summary()))

            print(f"OLS results for {x} and {y} saved to {ols_file}")