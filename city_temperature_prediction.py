import matplotlib.pyplot as plt
import pandas as pd
from house_price_prediction import plot_bar_plots
from sklearn.model_selection import train_test_split
from polynomial_fitting import PolynomialFitting
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def load_data(filename: str) -> pd.DataFrame:
  """
  Load city daily temperature dataset and preprocess data.
  Parameters
  ----------
  filename: str
      Path to house prices dataset

  Returns
  -------
  Design matrix and response vector (Temp)
  """
  df = pd.read_csv(filename, parse_dates=['Date'])

  df = df.dropna()
  df = df[df['Temp'] >= -60]  # Assuming under -60 is an invalid temperature value
  scaler = MinMaxScaler()
  # df['Temp'] = scaler.fit_transform(df['Temp'].values.reshape(-1, 1))
  # df['Temp'] = (df['Temp'] - df['Temp'].mean()) / df['Temp'].std()
  df['DayOfYear'] = df['Date'].dt.dayofyear

  df['DayOfYear'] = (df['DayOfYear'] - df['DayOfYear'].mean()) / df['DayOfYear'].std()
  return df


def monthly_error_bar(df, country):
  plt.figure(figsize=(10, 6))
  df_country = df[df['Country'] == country]
  df_grouped = df_country.groupby('Month').agg({'Temp': ['mean', 'std']})
  df_grouped.columns = ['Temp_mean', 'Temp_std']
  plt.errorbar(df_grouped.index, df_grouped['Temp_mean'], yerr=df_grouped['Temp_std'], label=country, capsize=5)
  plt.xlabel('Month')
  plt.ylabel('Temperature (°C)')
  plt.title('Average Monthly Temperature by Country')
  plt.legend()
  plt.grid(True)
  plt.savefig('saved_images/monthly_error_bar.jpg')


def dayofyear_temp_scatterplot(df, country):
  df_country = df[df['Country'] == country]
  plt.figure(figsize=(10, 6))
  for year in df_country['Year'].unique():
    df_year = df_country[df_country['Year'] == year]
    plt.scatter(df_year['DayOfYear'], df_year['Temp'], label=str(year), alpha=0.5)
  plt.xlabel('Day of Year')
  plt.ylabel('Temperature (°C)')
  plt.title(f'Temperature in {country} by Day of Year')
  plt.legend(title='Year')
  plt.grid(True)
  plt.savefig(f"saved_images/temp_day_of_year_all.jpg")


def plot_avg_temp_by_country_and_month(df: pd.DataFrame) -> None:
  """
  Group by Country and Month, calculate the average and standard deviation of temperatures,
  and plot the results with error bars.

  Parameters
  ----------
  df : pd.DataFrame
      DataFrame containing the city temperature data.
  """
  # Group by Country and Month and calculate the mean and standard deviation of temperatures
  grouped = df.groupby(['Country', 'Month']).agg({'Temp': ['mean', 'std']}).reset_index()
  grouped.columns = ['Country', 'Month', 'Temp_mean', 'Temp_std']

  # Plotting the results
  plt.figure(figsize=(12, 8))

  for country in grouped['Country'].unique():
    country_data = grouped[grouped['Country'] == country]
    plt.errorbar(country_data['Month'], country_data['Temp_mean'],
                 yerr=country_data['Temp_std'], label=country, capsize=5, marker='o')

  plt.xlabel('Month')
  plt.ylabel('Temperature (°C)')
  plt.title('Average Monthly Temperature by Country with Standard Deviation')
  plt.legend(title='Country')
  plt.grid(True)
  plt.savefig(f"saved_images/countries_avg_temp.jpg")


def perform_polynomial_fitting(df: pd.DataFrame, country: str, plot_polynomial: bool = False) -> None:
  """
  Perform polynomial fitting on the dataset for a specific country.

  Parameters
  ----------
  df : pd.DataFrame
      DataFrame containing the city temperature data.
  country : str
      The country for which to perform the polynomial fitting.
  plot_polynomial : bool, default=False
      Whether to plot the fitted polynomial.
  """
  # Filter data for the specified country
  df_country = df[df['Country'] == country]

  # Randomly split the dataset into training set (75%) and test set (25%)
  X = df_country['DayOfYear'].to_numpy()
  y = df_country['Temp'].to_numpy()
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

  degrees = range(1, 11)
  losses = []

  # Fit polynomial models and record the loss
  for k in degrees:
    model = PolynomialFitting(k)
    model.fit(X_train, y_train)
    loss = round(model.loss(X_test, y_test), 2)
    losses.append(loss)
    # print(f"Degree {k}: Test Error = {loss}")

    # Plot the fitted polynomial if plot_polynomial is True
    if plot_polynomial:
      X_vals = np.linspace(min(X), max(X), 500)
      y_vals = model.predict(X_vals)

      plt.figure(figsize=(10, 6))
      plt.scatter(X_train, y_train, color='green', label='Train Data')
      plt.scatter(X_test, y_test, color='blue', label='Test Data')
      plt.plot(X_vals, y_vals, color='red', label=f'Polynomial Degree {k}')
      plt.xlabel('Day of Year')
      plt.ylabel('Temperature (°C)')
      plt.title(f'Polynomial Degree {k} Fit for {country}')
      plt.legend()
      plt.grid(True)
      plt.savefig(f"saved_images/polynomial_fit_degree_{k}.jpg")
      plt.close()

  # Plot the test error for each degree
  plt.figure(figsize=(10, 6))
  plt.bar(degrees, losses, color='skyblue')
  plt.xlabel('Polynomial Degree')
  plt.ylabel('Mean Squared Error (Rounded to 2 Decimal Places)')
  plt.title(f'Test Error for Different Polynomial Degrees in {country}')
  plt.grid(True)
  plt.savefig(f"saved_images/best_fit_calculation.jpg")



def get_fit_model(df: pd.DataFrame, country: str, k: int) -> PolynomialFitting:
  """
  Fit a polynomial model for a specific country.

  Parameters
  ----------
  df : pd.DataFrame
      DataFrame containing the city temperature data.
  country : str
      The country for which to fit the polynomial model.
  k : int
      The degree of the polynomial model.

  Returns
  -------
  model : PolynomialFitting
      The fitted polynomial model.
  """
  df_country = df[df['Country'] == country]
  X = df_country['DayOfYear'].to_numpy()
  y = df_country['Temp'].to_numpy()
  model = PolynomialFitting(k)
  model.fit(X, y)
  return model


def evaluate_model_on_countries(df: pd.DataFrame, model: PolynomialFitting, exclude_country: str) -> pd.DataFrame:
  """
  Evaluate the model on different countries and compute the mean squared error.

  Parameters
  ----------
  df : pd.DataFrame
      DataFrame containing the city temperature data.
  model : PolynomialFitting
      The fitted polynomial model.
  exclude_country : str
      The country to exclude from the evaluation.

  Returns
  -------
  results : pd.DataFrame
      DataFrame containing the mean squared error for each country.
  """
  countries = df['Country'].unique()
  errors = []

  for country in countries:
    if country == exclude_country:
      continue
    df_country = df[df['Country'] == country]
    X = df_country['DayOfYear'].to_numpy()
    y = df_country['Temp'].to_numpy()
    error = model.loss(X, y)
    errors.append({'Country': country, 'Error': error})
  errors = pd.DataFrame(errors)

  plt.figure(figsize=(10, 6))
  plt.bar(errors['Country'], errors['Error'], color='skyblue')
  plt.xlabel('Country')
  plt.ylabel('Mean Squared Error')
  plt.title("Model Error for Different Countries Using Polynomial Fit for Israel (k=3)")
  plt.grid(True)
  plt.savefig(f"saved_images/MSE_each_country.jpg")


if __name__ == '__main__':
  # Question 2 - Load and preprocessing of city temperature dataset
  df = load_data("city_temperature.csv")
  # plot_bar_plots(df)

  # Question 3 - Exploring data for specific country
  dayofyear_temp_scatterplot(df, 'Israel')
  monthly_error_bar(df, 'Israel')

  # Question 4 - Exploring differences between countries
  plot_avg_temp_by_country_and_month(df)

  # Question 5 - Fitting model for different values of `k`
  perform_polynomial_fitting(df, 'Israel', True)

  # Question 6 - Evaluating fitted model on different countries
  model_israel = get_fit_model(df, 'Israel', k=3)
  evaluate_model_on_countries(df, model_israel, "Israel")
