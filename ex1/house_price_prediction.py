import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import NoReturn
from linear_regression import LinearRegression  # Ensure this file is accessible


def preprocess_train(X: pd.DataFrame, y: pd.Series) -> (pd.DataFrame, pd.Series):
    """
    Preprocess training data.

    Parameters
    ----------
    X: pd.DataFrame
        The input features.
    y: pd.Series, optional
        The target variable (house prices).

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series or None]
        A tuple containing the preprocessed features and the target variable.
    """
    combined_data = X.copy()
    if y is not None:
        combined_data['price'] = y

    # Remove missing values and duplicates
    combined_data = combined_data.dropna().drop_duplicates()

    # Drop unnecessary columns
    columns_to_drop = ["id", "lat", "long", "date"]
    combined_data = combined_data.drop(columns=columns_to_drop)

    # Filter out invalid values
    valid_value_conditions = {
        "sqft_living": (combined_data["sqft_living"] > 0),
        "sqft_lot": (combined_data["sqft_lot"] > 0),
        "sqft_above": (combined_data["sqft_above"] > 0),
        "yr_built": (combined_data["yr_built"] > 0),
        "bathrooms": (combined_data["bathrooms"] >= 0),
        "floors": (combined_data["floors"] >= 0),
        "sqft_basement": (combined_data["sqft_basement"] >= 0),
        "yr_renovated": (combined_data["yr_renovated"] >= 0),
        "waterfront": (combined_data["waterfront"].isin([0, 1])),
        "view": (combined_data["view"].isin(range(5))),
        "condition": (combined_data["condition"].isin(range(1, 6))),
        "grade": (combined_data["grade"].isin(range(1, 15)))
    }

    for col, condition in valid_value_conditions.items():
        combined_data = combined_data[condition]

    # Feature engineering for renovation and decade built
    combined_data["recently_renovated"] = np.where(combined_data["yr_renovated"] >= np.percentile(combined_data["yr_renovated"].unique(), 70), 1, 0)
    combined_data = combined_data.drop("yr_renovated", axis=1)

    combined_data["decade_built"] = (combined_data["yr_built"] // 10).astype(int)
    combined_data = combined_data.drop("yr_built", axis=1)

    # One-hot encoding for categorical variables
    combined_data = pd.get_dummies(combined_data, columns=["decade_built"], prefix='decade_built')

    # Remove outliers
    combined_data = combined_data[combined_data["bedrooms"] < 20]
    combined_data = combined_data[combined_data["sqft_lot"] < 1250000]

    if "price" in combined_data.columns:
        combined_data = combined_data[combined_data["price"] > 0]
        X_processed = combined_data.drop("price", axis=1)
        y_processed = combined_data["price"]
    else:
        X_processed = combined_data
        y_processed = None

    return X_processed, y_processed


def preprocess_test(X: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess test data.
    You are not allowed to remove rows from X, but only edit its columns.

    Parameters
    ----------
    X: pd.DataFrame
        The input data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    # Add new features that were added in the training preprocessing
    X["recently_renovated"] = np.where(X["yr_renovated"] >= np.percentile(X["yr_renovated"].unique(), 70), 1, 0)
    X["decade_built"] = (X["yr_built"] // 10).astype(int)

    # Drop columns not needed
    X = X.drop(columns=["id", "lat", "long", "date", "yr_renovated", "yr_built"], errors='ignore')

    # One-hot encoding for categorical variables
    X = pd.get_dummies(X, columns=["decade_built"], prefix='decade_built')

    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_directory: str = ".") -> NoReturn:
    """
     Generate scatter plots between each feature and the target variable, with Pearson correlation.

     Parameters
     ----------
     X: pd.DataFrame
         DataFrame containing the features.
     y: pd.Series
         Series containing the target variable (house prices).
     output_directory: str, default "."
         Directory where the plots will be saved.
     """
    # Exclude one-hot encoded columns from the analysis
    numeric_features = X.loc[:, ~X.columns.str.startswith('decade_built_')]

    for feature in numeric_features.columns:
        # Calculate the Pearson correlation coefficient
        feature_values = X[feature]
        correlation_matrix = np.corrcoef(feature_values, y)
        correlation = correlation_matrix[0, 1]

        # Create scatter plot with a trendline
        plt.figure()
        plt.scatter(feature_values, y, alpha=0.5)
        plt.title(f'{feature} vs. Price\nPearson Correlation: {correlation:.2f}')
        plt.xlabel(f'{feature}')
        plt.ylabel('Price')

        # Save the plot as a PNG file
        plot_filename = f'{output_directory}/{feature}_correlation.png'
        plt.savefig(plot_filename)
        plt.close()


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Question 3 - Preprocessing of training data
    X_train, y_train = preprocess_train(X_train, y_train)


    # Question 4 - Feature evaluation of training data with respect to response
    feature_evaluation(X_train, y_train)

    # # Question 5 - Preprocessing of test data
    X_test = preprocess_test(X_test)

    # Question 6 - Fit model over increasing percentages of the overall training data
    percentages = np.arange(10, 101)
    mean_losses = []
    loss_variances = []

    for percentage in percentages:
        loss_values = []
        for _ in range(10):
            sample_X_train = X_train.sample(frac=percentage / 100, random_state=None)
            sample_y_train = y_train[sample_X_train.index]
            linear_model = LinearRegression(include_intercept=True)
            linear_model.fit(sample_X_train, sample_y_train)
            test_loss = linear_model.loss(X_test, y_test)
            loss_values.append(test_loss)
        mean_losses.append(np.mean(loss_values))
        loss_variances.append(np.var(loss_values))

    # Plot average loss as a function of training size with error ribbon
    plt.figure()
    plt.plot(percentages, mean_losses, label='Average Loss')
    plt.fill_between(percentages, np.array(mean_losses) - 2 * np.sqrt(loss_variances),
                     np.array(mean_losses) + 2 * np.sqrt(loss_variances), color='blue', alpha=0.2,
                     label='Confidence Interval')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Average Loss')
    plt.title('Average Loss vs. Training Size')
    plt.legend()
    plt.savefig('loss_vs_training_size.png')
