import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split
from typing import NoReturn
import math
from linear_regression import LinearRegression


def preprocess_train(X: pd.DataFrame, y: pd.Series) -> (pd.DataFrame, pd.Series):
    """
    Preprocess the training data.

    Parameters
    ----------
    X: pd.DataFrame
        The DataFrame containing the features.
    y: pd.Series
        The Series containing the response variable.

    Returns
    -------
    X: pd.DataFrame
        The preprocessed features DataFrame.
    y: pd.Series
        The preprocessed response Series.
    """
    X = pd.concat([X, y.rename("price")], axis=1)
    X.dropna(inplace=True)

    # Add year, month, and day columns
    X['date'] = pd.to_datetime(X['date'], format='%Y%m%dT000000')
    X['year'] = X['date'].dt.year
    X['month'] = X['date'].dt.month
    X['day'] = X['date'].dt.day

    # Remove the date column
    X.drop(columns=['date'], inplace=True)

    # Remove rows where yr_renovated is lower than yr_built (except when yr_renovated is 0)
    mask = (X['yr_renovated'] != 0) & (X['yr_renovated'] < X['yr_built'])
    if mask.any():
        print("Rows with yr_renovated lower than yr_built (except 0):")
        print(X[mask])
        X = X[~mask]

    # Replace yr_renovated with binary indicator
    X['is_renovated'] = X['yr_renovated'].map(lambda x: 0 if x == 0 else 1)
    X['has_basement'] = X['sqft_basement'].map(lambda x: 0 if x == 0 else 1)

    X.drop(columns=['yr_renovated'], inplace=True)
    X.drop(columns=['sqft_basement'], inplace=True)

    # Remove the id column
    if 'id' in X.columns:
        X.drop(columns=['id'], inplace=True)

    # Remove rows with illogically low sqft_living values
    low_sqft_living_threshold = 100  # Define a threshold for illogical values
    low_sqft_living_mask = X['sqft_living'] < low_sqft_living_threshold
    if low_sqft_living_mask.any():
        print("Rows with illogically low sqft_living values:")
        print(X[low_sqft_living_mask])
        X = X[~low_sqft_living_mask]

    # Add age column
    X['age'] = X['year'] - X['yr_built']

    # Separate features and target variable
    y = X.pop("price")

    return X, y


def preprocess_test(X: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the test data.

    Parameters
    ----------
    X: pd.DataFrame
        The DataFrame containing the test features.

    Returns
    -------
    pd.DataFrame
        The preprocessed features DataFrame.
    """
    # Impute missing values
    X = X.replace([np.inf, -np.inf], np.nan)
    X.ffill(inplace=True)
    X.bfill(inplace=True)

    # Add year, month, and day columns
    X['date'] = pd.to_datetime(X['date'], format='%Y%m%dT000000')
    X['year'] = X['date'].dt.year
    X['month'] = X['date'].dt.month
    X['day'] = X['date'].dt.day

    # Remove the date column
    X.drop(columns=['date'], inplace=True)

    # Replace yr_renovated with binary indicator
    X['is_renovated'] = X['yr_renovated'].map(lambda x: 0 if x == 0 else 1)
    X.drop(columns=['yr_renovated'], inplace=True)

    # Replace sqft_basement with binary indicator
    X['has_basement'] = X['sqft_basement'].map(lambda x: 0 if x == 0 else 1)
    X.drop(columns=['sqft_basement'], inplace=True)

    # Remove the id column
    if 'id' in X.columns:
        X.drop(columns=['id'], inplace=True)

    # Add age column
    X['age'] = X['year'] - X['yr_built']

    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = "saved_images/feature_evaluation.jpg") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    num_features = X.shape[1]
    num_cols = 4
    num_rows = math.ceil(num_features / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    axes = axes.flatten()

    for i, col in enumerate(X.columns):
        # Calculate Pearson correlation manually
        cov = np.cov(X[col], y)[0, 1]
        std_x = np.std(X[col], ddof=1)
        std_y = np.std(y, ddof=1)
        pearson_corr = cov / (std_x * std_y)

        # Create scatter plot
        axes[i].scatter(X[col], y, alpha=0.5)
        axes[i].set_title(f'{col}\nPearson Correlation: {pearson_corr:.2f}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Response')

    # Hide any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_bar_plots(df: pd.DataFrame, figsize: tuple = (20, 20)) -> None:
    """
    Plot bar plots for all columns in the DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame containing the data.

    figsize: tuple, default=(20, 20)
        Size of the entire figure.
    """
    num_columns = len(df.columns)
    num_rows = int(np.ceil(np.sqrt(num_columns)))  # Calculate the number of rows needed for a square layout
    num_cols = int(np.ceil(num_columns / num_rows))  # Calculate the number of columns needed for a square layout

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()  # Flatten the array of axes for easy iteration

    for i, col in enumerate(df.columns):
        unique_values = df[col].nunique()
        title = f"{col} (Unique values Num: {unique_values})"
        if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
            df[col].value_counts().plot(kind='bar', ax=axes[i], edgecolor='black')
        else:
            df[col].plot(kind='hist', ax=axes[i], bins=30, edgecolor='black')
        axes[i].set_title(title)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_points_with_clustering(df: pd.DataFrame, lat_col: str, lon_col: str, map_location: tuple = (0, 0), zoom_start: int = 2) -> folium.Map:
    """
    Plot points on a map using latitude and longitude columns with clustering.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame containing the data.
    lat_col: str
        The name of the column containing the latitude values.
    lon_col: str
        The name of the column containing the longitude values.
    map_location: tuple, default=(0, 0)
        Initial location to center the map.
    zoom_start: int, default=2
        Initial zoom level for the map.

    Returns
    -------
    folium.Map
        A folium map with the points plotted using clustering.
    """
    # Create a folium map centered at the given location
    map_ = folium.Map(location=map_location, zoom_start=zoom_start)

    # Create a MarkerCluster object
    marker_cluster = MarkerCluster().add_to(map_)

    # Add points to the cluster
    for _, row in df.iterrows():
        folium.Marker(location=[row[lat_col], row[lon_col]]).add_to(marker_cluster)
    map_.save("map_with_clustering.html")
    return map_


def plot_feature_correlations(X: pd.DataFrame, y: pd.Series,output_path: str = "saved_images/pearson_chart.jpg") -> None:
    """
    Plot the Pearson correlation between each feature in X and the response y.

    Parameters
    ----------
    X: pd.DataFrame
        The DataFrame containing the features.
    y: pd.Series
        The Series containing the response variable.

    Returns
    -------
    None
    """
    correlations = {}

    # Calculate Pearson correlation for each feature
    for i, col in enumerate(X.columns):
        # Calculate Pearson correlation manually
        cov = np.cov(X[col], y)[0, 1]
        std_x = np.std(X[col], ddof=1)
        std_y = np.std(y, ddof=1)
        pearson_corr = cov / (std_x * std_y)

        correlations[col] = pearson_corr

    # Convert the correlations dictionary to a DataFrame
    corr_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Pearson Correlation'])

    # Plot the correlations
    plt.figure(figsize=(12, 8))
    plt.barh(corr_df['Feature'], corr_df['Pearson Correlation'], color='skyblue')
    plt.xlabel('Pearson Correlation')
    plt.title('Pearson Correlation between Features and Response')
    plt.grid(True)
    plt.savefig(output_path)


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)

    # Question 4 - Feature evaluation of train dataset with respect to response
    # feature_evaluation(X_train, y_train)
    # plot_feature_correlations(X_train, y_train)

    # Question 5 - preprocess the test data
    X_test = preprocess_test(X_test)

    # Question 6 - Fit model over increasing percentages of the overall training data
    reps = 5
    percentages = np.arange(10, 101, 1)
    mean_losses = []
    std_losses = []

    for p in percentages:
        losses = []
        for _ in range(reps):
            # 1) Sample p% of the overall training data
            X_sample = X_train.sample(frac=p / 100, random_state=None)
            y_sample = y_train.loc[X_sample.index]

            # 2) Fit linear model (including intercept) over sampled set
            model = LinearRegression(include_intercept=True)
            model.fit(X_sample.to_numpy(), y_sample.to_numpy())

            # 3) Test fitted model over test set
            loss = model.loss(X_test.to_numpy(), y_test.to_numpy())
            losses.append(loss)

        # 4) Store average and variance of loss over test set
        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))

    # Plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    mean_losses = np.array(mean_losses)
    std_losses = np.array(std_losses)
    lower_bound = mean_losses - 2 * std_losses
    upper_bound = mean_losses + 2 * std_losses

    plt.figure(figsize=(10, 6))
    plt.plot(percentages, mean_losses, label='Mean Loss')
    plt.fill_between(percentages, lower_bound, upper_bound, color='b', alpha=0.2, label='Confidence Interval')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Mean Loss as a Function of Training Size\n {reps} repetitions')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"saved_images/training loss {reps} reps.jpg")

    print("Done")
