import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Define Euclidean distance
def euclidean_distance(X1, X2):
    return np.sqrt(np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2))

# Define Manhattan distance
def manhattan_distance(X1, X2):
    return np.sum(np.abs(X1[:, np.newaxis] - X2), axis=2)

# Define Cosine distance
def cosine_distance(X1, X2):
    dot_product = np.dot(X1, X2.T)
    norm_X1 = np.linalg.norm(X1, axis=1, keepdims=True)
    norm_X2 = np.linalg.norm(X2, axis=1, keepdims=True)
    return 1 - (dot_product / (norm_X1 * norm_X2.T))

# KNN class
class InitialKNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def calculate_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'cosine':
            dot_product = np.dot(x1, x2)
            norm_x1 = np.linalg.norm(x1)
            norm_x2 = np.linalg.norm(x2)
            return 1 - (dot_product / (norm_x1 * norm_x2))

    def _predict(self, x):
        distances = [self.calculate_distance(x, x_train) for x_train in self.X_train]
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_nearest_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        return [self._predict(x) for x in X]

    def validate(self, X_val, Y_val):
        y_pred = self.predict(X_val)
        metrics = CustomMetrics()
        return metrics.validate(Y_val, y_pred)

# Custom metrics class
class CustomMetrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
        return correct / len(y_true)

    @staticmethod
    def precision_recall_f1(y_true, y_pred):
        true_positive = Counter()
        false_positive = Counter()
        false_negative = Counter()
        
        for y_t, y_p in zip(y_true, y_pred):
            if y_t == y_p:
                true_positive[y_t] += 1
            else:
                false_positive[y_p] += 1
                false_negative[y_t] += 1
        
        precision = {}
        recall = {}
        f1_score = {}
        
        for label in set(y_true):
            precision[label] = true_positive[label] / (true_positive[label] + false_positive[label]) if (true_positive[label] + false_positive[label]) > 0 else 0
            recall[label] = true_positive[label] / (true_positive[label] + false_negative[label]) if (true_positive[label] + false_negative[label]) > 0 else 0
            f1_score[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label]) if (precision[label] + recall[label]) > 0 else 0
        
        precision_macro = np.mean(list(precision.values()))
        recall_macro = np.mean(list(recall.values()))
        f1_macro = np.mean(list(f1_score.values()))
        
        precision_micro = sum(true_positive.values()) / (sum(true_positive.values()) + sum(false_positive.values())) if (sum(true_positive.values()) + sum(false_positive.values())) > 0 else 0
        recall_micro = sum(true_positive.values()) / (sum(true_positive.values()) + sum(false_negative.values())) if (sum(true_positive.values()) + sum(false_negative.values())) > 0 else 0
        f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0
        
        return precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro

    def validate(self, y_true, y_pred):
        accuracy = self.accuracy(y_true, y_pred)
        precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro = self.precision_recall_f1(y_true, y_pred)
        return {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro
        }

# Polynomial Regression class
class PolynomialRegression:
    def __init__(self, degree):
        self.degree = degree
        self.coefficients = None

    def fit(self, X, y):
        X_poly = self._create_polynomial_features(X)
        self.coefficients = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)

    def predict(self, X):
        X_poly = self._create_polynomial_features(X)
        return X_poly.dot(self.coefficients)

    def _create_polynomial_features(self, X):
        X_poly = np.ones((X.shape[0], 1))
        for i in range(1, self.degree + 1):
            X_poly = np.c_[X_poly, X**i]
        return X_poly

# Regularized Polynomial Regression class
class RegularizedPolynomialRegression:
    def __init__(self, degree, regularization_type=None, lambda_=0.1):
        self.degree = degree
        self.regularization_type = regularization_type
        self.lambda_ = lambda_
        self.coefficients = None

    def fit(self, X, y):
        X_poly = self._create_polynomial_features(X)
        if self.regularization_type == 'L1':
            self.coefficients = self._lasso_regression(X_poly, y)
        elif self.regularization_type == 'L2':
            self.coefficients = self._ridge_regression(X_poly, y)
        else:
            self.coefficients = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)

    def predict(self, X):
        X_poly = self._create_polynomial_features(X)
        return X_poly.dot(self.coefficients)

    def _create_polynomial_features(self, X):
        X_poly = np.ones((X.shape[0], 1))
        for i in range(1, self.degree + 1):
            X_poly = np.c_[X_poly, X**i]
        return X_poly

    def _ridge_regression(self, X_poly, y):
        I = np.eye(X_poly.shape[1])
        I[0, 0] = 0
        return np.linalg.inv(X_poly.T.dot(X_poly) + self.lambda_ * I).dot(X_poly.T).dot(y)

    def _lasso_regression(self, X_poly, y):
        from scipy.optimize import minimize

        def lasso_loss(coefficients):
            return np.sum((X_poly.dot(coefficients) - y) ** 2) + self.lambda_ * np.sum(np.abs(coefficients))

        result = minimize(lasso_loss, np.zeros(X_poly.shape[1]), method='L-BFGS-B')
        return result.x

# Linear Regression class
class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, lambda_=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_ = lambda_
        self.beta = None

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.beta = np.zeros(X.shape[1])

        for _ in range(self.epochs):
            predictions = X.dot(self.beta)
            errors = predictions - y
            gradient = (2 / X.shape[0]) * (X.T.dot(errors) + self.lambda_ * self.beta)
            self.beta -= self.learning_rate * gradient

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X.dot(self.beta)

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def variance(self, y_true, y_pred):
        return np.var(y_true - y_pred)

    def std_dev(self, y_true, y_pred):
        return np.std(y_true - y_pred)

# Load the dataset
data = pd.read_csv('linreg.csv')
X = data.iloc[:, 0].values
y = data.iloc[:, 1].values

# Shuffle the data
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Split the data into train (80%), validation (10%), and test (10%) sets
split1 = int(0.8 * len(X))
split2 = int(0.9 * len(X))

X_train, y
