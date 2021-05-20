import numpy as np
from regression import Regression


class LogisticRegression(Regression):

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_fn(self, X, y):
        a = self.sigmoid(self.predict(X))
        cost = np.sum(-y * np.log(a) - (1 - y) * np.log(1 - a))
        return cost

    def gradient(self, X, y):
        a = self.sigmoid(self.predict(X))
        return np.dot(X.T, (a - y))
