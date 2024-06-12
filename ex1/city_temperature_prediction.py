import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from typing import NoReturn
from linear_regression import LinearRegression
from polynomial_fitting import PolynomialFitting

def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to city temperature dataset

    Returns
    -------
    df: pd.DataFrame
        Preprocessed DataFrame
    """
    # Load dataset with date parsing
    df = pd.read_csv(filename, parse_dates=['Date'])

    # Remove invalid data (e.g., temperature below -50 or above 50)
    # df = df[(df['Temp'] > -50) & (df['Temp'] < 50)]
    df = df[(df['Temp'] > 0)]

    # Add 'DayOfYear' column
    df['DayOfYear'] = df['Date'].dt.dayofyear

    return df


def plot_temperature_by_day_of_year(df: pd.DataFrame) -> None:
    """
    Plot average daily temperature as a function of DayOfYear, color-coded by year.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the filtered data
    """
    df_israel = df[df['Country'] == 'Israel']

    # Plot scatter plot
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(df_israel['DayOfYear'], df_israel['Temp'], c=df_israel['Year'], cmap='tab20', alpha=0.6)
    plt.colorbar(scatter, label='Year')
    plt.xlabel('Day of Year')
    plt.ylabel('Temperature (°C)')
    plt.title('Average Daily Temperature in Israel by Day of Year')
    plt.savefig('israel_daily_temperatures.png')
    plt.close()


def plot_temperature_by_month(df: pd.DataFrame) -> None:
    """
    Plot standard deviation of daily temperatures by month.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the filtered data
    """
    df_israel = df[df['Country'] == 'Israel']

    # Group by Month and calculate standard deviation
    std_by_month = df_israel.groupby('Month')['Temp'].agg('std').reset_index()

    # Plot bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Month', y='Temp', data=std_by_month, palette='viridis')
    plt.xlabel('Month')
    plt.ylabel('Standard Deviation of Temperature (°C)')
    plt.title('Standard Deviation of Daily Temperatures by Month in Israel')
    plt.savefig('israel_monthly_std_temp.png')
    plt.close()

def plot_avg_temp_with_error_bars(df: pd.DataFrame) -> None:
    """
    Plot average monthly temperature with error bars by country.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the data
    """
    # Group by Country and Month, and calculate mean and standard deviation
    grouped = df.groupby(['Country', 'Month']).agg({'Temp': ['mean', 'std']}).reset_index()
    grouped.columns = ['Country', 'Month', 'AvgTemp', 'StdTemp']

    # Plot using plotly
    fig = px.line(grouped, x='Month', y='AvgTemp', color='Country', error_y='StdTemp',
                  labels={'AvgTemp': 'Average Temperature (°C)', 'Month': 'Month'},
                  title='Average Monthly Temperature with Error Bars by Country')
    fig.write_image("average_monthly_temperatures_by_country.png")


def fit_polynomial_models(df: pd.DataFrame) -> None:
    """
    Fit polynomial models of varying degrees and evaluate their performance on the test set.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the data from Israel
    """
    df_israel = df[df['Country'] == 'Israel']

    # Split the data into training and test sets
    X = df_israel['DayOfYear'].values.reshape(-1, 1)
    y = df_israel['Temp']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    test_errors = []

    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(X_train, y_train)
        loss = model.loss(X_test, y_test)
        test_errors.append(round(loss, 2))
        # print(f'Test error for k={k}: {test_errors[-1]}')

    # Plot bar plot of test errors
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 11), test_errors, color='skyblue')
    plt.xlabel('Polynomial Degree (k)')
    plt.ylabel('Test Error (MSE)')
    plt.title('Test Error for Different Polynomial Degrees')
    plt.savefig('test_error_polynomial_degrees.png')
    plt.close()


def evaluate_model_on_countries(df: pd.DataFrame) -> None:
    """
    Evaluate the fitted polynomial model on different countries.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the data
    """
    # Filter data for Israel and split into training and test sets
    df_israel = df[df['Country'] == 'Israel']
    X_israel = df_israel['DayOfYear'].values.reshape(-1, 1)
    y_israel = df_israel['Temp']
    X_train_israel, X_test_israel, y_train_israel, y_test_israel = train_test_split(X_israel, y_israel, test_size=0.25,
                                                                                    random_state=42)

    # Fit the best polynomial model (assuming k=5 for this example)
    best_k = 5
    model = PolynomialFitting(best_k)
    model.fit(X_train_israel, y_train_israel)

    # Evaluate on other countries
    countries = df['Country'].unique()
    losses = []

    for country in countries:
        if country == 'Israel':
            continue
        df_country = df[df['Country'] == country]
        X_country = df_country['DayOfYear'].values.reshape(-1, 1)
        y_country = df_country['Temp']
        loss = model.loss(X_country, y_country)
        losses.append((country, round(loss, 2)))

    # Convert losses to DataFrame
    losses_df = pd.DataFrame(losses, columns=["Country", "Loss"])

    # Plot bar plot of losses
    plt.figure(figsize=(10, 6))
    plt.bar(losses_df['Country'], losses_df['Loss'], color='skyblue')
    plt.xlabel('Country')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss Over Countries for Model Fitted Over Israel')
    plt.savefig('loss_over_countries.png')
    plt.close()


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    plot_temperature_by_day_of_year(df)
    plot_temperature_by_month(df)


    # Question 4 - Exploring differences between countries
    plot_avg_temp_with_error_bars(df)


    # Question 5 - Fitting model for different values of `k`
    fit_polynomial_models(df)


    #
    # Question 6 - Evaluating fitted model on different countries
    evaluate_model_on_countries(df)

