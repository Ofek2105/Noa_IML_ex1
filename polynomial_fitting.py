import numpy as np
from linear_regression import LinearRegression
from typing import NoReturn


class PolynomialFitting(LinearRegression):
  """
  Polynomial Fitting using Least Squares estimation
  """

  def __init__(self, k: int):
    """
    Instantiate a polynomial fitting estimator

    Parameters
    ----------
    k : int
        Degree of polynomial to fit
    """
    super().__init__(include_intercept=False)
    self.k = k

  def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
    """
    Fit Least Squares model to polynomial transformed samples

    Parameters
    ----------
    X : ndarray of shape (n_samples,)
        Input data to fit an estimator for

    y : ndarray of shape (n_samples, )
        Responses of input data to fit to
    """
    X_transformed = self._transform(X)
    super(PolynomialFitting, self).fit(X_transformed, y)

  def predict(self, X: np.ndarray) -> np.ndarray:
    """
    Predict responses for given samples using fitted estimator

    Parameters
    ----------
    X : ndarray of shape (n_samples,)
        Input data to predict responses for

    Returns
    -------
    responses : ndarray of shape (n_samples, )
        Predicted responses of given samples
    """
    X_transformed = self._transform(X)
    return super(PolynomialFitting, self).predict(X_transformed)

  def loss(self, X: np.ndarray, y: np.ndarray) -> float:
    """
    Evaluate performance under MSE loss function

    Parameters
    ----------
    X : ndarray of shape (n_samples,)
        Test samples

    y : ndarray of shape (n_samples, )
        True labels of test samples

    Returns
    -------
    loss : float
        Performance under MSE loss function
    """
    X_transformed = self._transform(X)
    return LinearRegression.loss(self, X_transformed, y)

  def _transform(self, X: np.ndarray) -> np.ndarray:
    """
    Transform given input according to the univariate polynomial transformation

    Parameters
    ----------
    X: ndarray of shape (n_samples,)

    Returns
    -------
    transformed: ndarray of shape (n_samples, k+1)
        Vandermonde matrix of given samples up to degree k
    """
    if X.ndim == 1:
      return np.vander(X, N=self.k + 1, increasing=True)
    else:
      raise ValueError("X must be a one-dimensional array or sequence.")
