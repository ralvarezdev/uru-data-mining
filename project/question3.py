from project.load import load_transformed_dataset
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# What are the most relevant factors in predicting Economic Loss from accidents?

if __name__ == '__main__':
    print("--- QUESTION 3 ---\n")

    # Load the transformed dataset CSV file
    df = load_transformed_dataset()

    # Define features and target
    X = df.drop(columns=['Economic Loss'])
    y = df['Economic Loss']

    # Encode categorical variables
    X = pd.get_dummies(X)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')

    # Feature importance
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    print(feature_importance.sort_values(by='Importance', ascending=False))