import numpy as np

rng = np.random.default_rng(seed=42)


class ActivionFunctions:
    def __init__(self):
        pass

    @staticmethod
    def linear(x: np.float16) -> float:
        return x * rng.uniform(1.0, 100.0)

    @staticmethod
    def softmax(x, axis: int | None = None):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=axis)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def gelu(x):
        """Tanh approximation"""
        inside = 1 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        return 1 / 2 * x * inside
