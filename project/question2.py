from project.load import load_transformed_dataset
import statsmodels.api as sm

# How does Speed Limit and Traffic Volume affect the Number of Injuries and Number of Fatalities?

if __name__ == '__main__':
    print("--- QUESTION 2 ---\n")

    # Load the transformed dataset CSV file
    df = load_transformed_dataset()

    # Define features and target
    X = df[['Speed Limit', 'Traffic Volume']]
    y_injuries = df['Number of Injuries']
    y_fatalities = df['Number of Fatalities']

    # Add a constant to the model (intercept)
    X = sm.add_constant(X)

    # Fit the regression model for Number of Injuries
    model_injuries = sm.OLS(y_injuries, X).fit()
    print(model_injuries.summary())

    # Fit the regression model for Number of Fatalities
    model_fatalities = sm.OLS(y_fatalities, X).fit()
    print(model_fatalities.summary())