#Used LLM services  for polynomial regression
import numpy as np
# Polynomial Regression class for fitting and predicting polynomial models
class PolynomialRegression:
    def __init__(self, degree):
        self.degree = degree
        self.coefficients = None

    # Fit the polynomial regression model by calculating coefficients
    def fit(self, X, y):
        X_poly = self._create_polynomial_features(X)  # Create polynomial features
        self.coefficients = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)  # Calculate coefficients using normal equation

    # Predict using the fitted model
    def predict(self, X):
        X_poly = self._create_polynomial_features(X)  # Create polynomial features
        return X_poly.dot(self.coefficients)  # Return predictions

    # Create polynomial features up to the specified degree
    def _create_polynomial_features(self, X):
        X_poly = np.ones((X.shape[0], 1))  # Start with a column of ones for the intercept term
        for i in range(1, self.degree + 1):
            X_poly = np.c_[X_poly, X**i]  # Add polynomial features up to the specified degree
        return X_poly
#Used LLM SERVICES as reference for polynomial regression class
# Class for Polynomial Regression with Regularization (L1 or L2)
class RegularizedPolynomialRegression:
    def __init__(self, degree, regularization_type=None, lambda_=0.1):
        # Initialize the model with polynomial degree, regularization type (L1 or L2), and lambda (regularization strength)
        self.degree = degree
        self.regularization_type = regularization_type
        self.lambda_ = lambda_
        self.coefficients = None

    def fit(self, X, y):
        # Fit the model to the training data
        X_poly = self._create_polynomial_features(X)
        if self.regularization_type == 'L1':
            # L1 Regularization (Lasso Regression)
            self.coefficients = self._lasso_regression(X_poly, y)
        elif self.regularization_type == 'L2':
            # L2 Regularization (Ridge Regression)
            self.coefficients = self._ridge_regression(X_poly, y)
        else:
            # No Regularization (Standard Polynomial Regression)
            self.coefficients = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)

    def predict(self, X):
        # Predict the output for given input data
        X_poly = self._create_polynomial_features(X)
        return X_poly.dot(self.coefficients)

    def _create_polynomial_features(self, X):
        # Generate polynomial features (X^0, X^1, X^2, ..., X^degree)
        X_poly = np.ones((X.shape[0], 1))  # Start with a column of ones (X^0)
        for i in range(1, self.degree + 1):
            X_poly = np.c_[X_poly, X**i]  # Add columns for each degree of X
        return X_poly

    def _ridge_regression(self, X_poly, y):
        # Ridge Regression (L2 Regularization)
        I = np.eye(X_poly.shape[1])  # Identity matrix for regularization
        I[0, 0] = 0  # Do not regularize the bias term
        return np.linalg.inv(X_poly.T.dot(X_poly) + self.lambda_ * I).dot(X_poly.T).dot(y)

    def _lasso_regression(self, X_poly, y):
        # Lasso Regression (L1 Regularization)
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=self.lambda_, fit_intercept=True, max_iter=10000)
        lasso.fit(X_poly, y)
        return lasso.coef_
# Linear Regression class
#I have used LLM services for gradient descent code reference
class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, lambda_=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_ = lambda_  # Regularization parameter
        self.beta = None

    def fit(self, X, y):
        # Add intercept (column of ones) to X
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Initialize beta (coefficients) with zeros
        self.beta = np.zeros(X.shape[1])

        # Gradient Descent
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
X, y = shuffle(X, y, random_state=42)

# Split the data into train (80%), validation (10%), and test (10%) sets
split1 = int(0.8 * len(X))
split2 = int(0.9 * len(X))

X_train, y_train = X[:split1], y[:split1]
X_val, y_val = X[split1:split2], y[split1:split2]
X_test, y_test = X[split2:], y[split2:]

# Reshape X for single feature regression
X_train = X_train.reshape(-1, 1)
X_val = X_val.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Initialize and fit the model
model = LinearRegression(learning_rate=0.01, epochs=1000, lambda_=0)
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)


