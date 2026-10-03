import numpy as np


class EvaluationMetrics:
    def __init__(self, n: int, y, y_pred):
        self.n = n
        self.y = np.reshape(y, (-1, 1))  # actual value
        self.y_pred = np.reshape(y_pred, (-1, 1))  # predicted value

    def mae(self):
        """
        Measures the average of the residuals.
        -> residuals is the distance between the predicted
        point and the actual point, in this case the absolute diff.
        """
        return np.mean(np.abs(self.y - self.y_pred))

    def mse(self):
        """
        Measure the average of the square residuals
        (Variance -> how far the point is spread out from their average).
        """
        return np.mean(np.pow((self.y - self.y_pred), 2))

    def rmse(self):
        """
        It's de square root of the MSE and it
        measures the standard deviation of the residuals.
        """
        return np.sqrt(np.mean(np.pow((self.y - self.y_pred), 2)))

    def rsquared(self):
        """
        RSquared or Coefficient of Determination, shows the proportion
        of variance in the dependent variable (y) that can be explained by
        the independent variable (X). This tells us how well the data fit
        the model, rather than indicate the correctness.
        RSS (Residual sum of the squares) / TSS (Total sum of the squares)
        The results ranges from 0 to 1 and we often seek to achieve a R2
        closer to 1, but that's not a rule.
        """
        y_mean = np.mean(self.y)
        RSS = sum(np.pow((self.y_pred - self.y), 2))
        TSS = sum(np.pow((self.y - y_mean), 2))
        return 1 - RSS / TSS
