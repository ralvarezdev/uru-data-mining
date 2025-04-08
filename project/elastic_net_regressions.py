from project import ELASTIC_NET_REGRESSIONS_DIR, ELASTIC_NET_REGRESSIONS_FILE
from project.combinations import generate_combinations
from project.load import load_transformed_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
import os

# Constants
MAX_SIZE = 3

if __name__ == '__main__':
    print("--- ELASTIC NET REGRESSIONS ---\n")

    # Load the combinations of columns
    df = load_transformed_dataset()
    combinationsList=[]
    for i in range(2, MAX_SIZE + 1):
        combinationsList.append(generate_combinations(i))
        print(f"Generated {len(combinationsList[-1])} combinations of size {i}")

    # Verify that the Elastic Net Regressions directory exists, if not create it
    if not os.path.exists(ELASTIC_NET_REGRESSIONS_DIR):
        os.makedirs(ELASTIC_NET_REGRESSIONS_DIR)

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

            # ElasticNetCV with cross-validation
            elastic_net_cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=[0.01, 0.1, 1, 10, 100], cv=5,
                                          random_state=42)
            elastic_net_cv.fit(X_train, Y_train)

            # Optimal parameters
            optimal_alpha = elastic_net_cv.alpha_
            optimal_l1_ratio = elastic_net_cv.l1_ratio_

            # Evaluate the model
            r2 = elastic_net_cv.score(X_test, Y_test)

            # Save the combination if R^2
            r2_combinations[combination] = r2

    # Open the file in write mode
    with open(os.path.join(ELASTIC_NET_REGRESSIONS_DIR, ELASTIC_NET_REGRESSIONS_FILE), 'w') as file:
        file.write(str(r2_combinations))

    print(f"Ridge Regressions results saved to {ELASTIC_NET_REGRESSIONS_FILE}")