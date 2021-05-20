import numpy as np


class Regression:
    def __init__(self, learning_rate=0.01, max_it=1000):
        self.learning_rate = learning_rate
        self.max_it = max_it
        self.W = None

    def cost_fn(self, X, y):
        prediction = self.predict(X)
        error = prediction - y
        m = y.size
        cost = 1 / (2 * m) * np.dot(error.T, error)
        return cost

    def gradient(self, X, y):
        prediction = self.predict(X)
        error = prediction - y
        m = y.size
        grad = (1 / m) * np.dot(X.T, error)
        return grad

    def predict(self, X):
        return np.dot(X, self.W)

    def fit(self, X, y):
        costs = []
        self.W = np.random.rand(X.shape[1])
        i = 0
        while i < self.max_it:
            costs.append(self.cost_fn(X, y))
            grads = self.gradient(X, y)
            self.W = self.W - self.learning_rate * grads
            i += 1
