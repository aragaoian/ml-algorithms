import numpy as np
from utils.weights import Weights


class Regression:
    def __init__(self, n_iter: int, l_rate: float):
        self.n_iter = n_iter
        self.l_rate = l_rate
        self.w = None
        self.b = 0
        self.error = 0

    def regularize(self, lamb: float = 0):
        return {
            None: 0,
            "L1": lamb * np.sign(self.w),
            "L2": 2 * lamb * self.w,
        }

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def fit(self, X, y, reg_factor: str = None):
        n_samples, n_features = X.shape
        if self.w is None:  # persisting weights
            self.w = Weights(n_features).random_uniform()
        y = np.reshape(y, (-1, 1))  # transform an array to a column vector
        regularization = self.regularize()

        for _ in range(self.n_iter):
            y_pred = self.predict(X)
            self.error = y - y_pred
            grad_w = -(np.dot(X.T, self.error) / n_samples) + regularization.get(
                reg_factor, 0
            )
            grad_b = -(np.sum(self.error)) / n_samples
            self.w -= self.l_rate * grad_w  # l_rate * (grad_w + regularization_factor)
            self.b -= self.l_rate * grad_b  # l_rate * grad_b


class LinearRegression(Regression):
    def __init__(self, n_iter: int, l_rate: float):
        super().__init__(n_iter, l_rate)

    def printWeights(self):
        print(self.w, self.w.shape)

    def gradientDescent(self, X, y):
        super().fit(X, y)

    def ordinaryleastSquares(self, X, y):
        """
        This algorithm works by making the total of the squares of the errors as
        small as possible. However, this model is sensitive to outliers, adding bias
        and pulling the line towards it.
        """

        y = y.reshape(-1, 1)

        x_mean = np.mean(X)
        y_mean = np.mean(y)

        self.w = sum((X - x_mean) * (y - y_mean)) / sum(np.pow(X - x_mean, 2))
        self.b = y_mean - self.w * x_mean

    def moorePenroseLeastSquares(self, X, y):
        y = y.reshape(-1, 1)
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term

        # Compute optimal weights
        theta = np.linalg.inv(X.T @ X) @ X.T @ y

        self.b = theta[0]
        self.w = theta[1:]
