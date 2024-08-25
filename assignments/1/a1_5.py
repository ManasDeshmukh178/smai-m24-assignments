import numpy as np
import matplotlib.pyplot as plt
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

# Load the data from 'regularisation.csv'
data = np.loadtxt('regularisation.csv', delimiter=',', skiprows=1)
X = data[:, 0]  # Feature (independent variable)
y = data[:, 1]  # Target (dependent variable)

# Shuffle the data to ensure randomness
np.random.seed(0)
indices = np.random.permutation(len(X))
X, y = X[indices], y[indices]

# Split the data into train (80%) and test (20%) sets
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Visualize the training dataset
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training Data')
plt.legend()
plt.show()

# Fit polynomials and plot the results
degrees = range(1, 21)  # Degrees 1 to 20
for i, degree in enumerate(degrees):
    # Iterate through regularization types: None, L1, and L2
    for j, reg_type in enumerate([None, 'L1', 'L2']):
        model = RegularizedPolynomialRegression(degree=degree, regularization_type=reg_type, lambda_=0.1)
        model.fit(X_train, y_train)

        # Predictions on training and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Compute metrics: MSE, standard deviation, and variance for both train and test sets
        mse_train = np.mean((y_train - y_train_pred) ** 2)
        mse_test = np.mean((y_test - y_test_pred) ** 2)
        std_train = np.std(y_train_pred)
        var_train = np.var(y_train_pred)
        std_test = np.std(y_test_pred)
        var_test = np.var(y_test_pred)

        # Print metrics to the terminal
        print(f"Degree {degree}, Regularization {reg_type}:")
        print(f"  MSE on Training Set: {mse_train}")
        print(f"  Standard Deviation on Training Set: {std_train}")
        print(f"  Variance on Training Set: {var_train}")
        print(f"  MSE on Test Set: {mse_test}")
        print(f"  Standard Deviation on Test Set: {std_test}")
        print(f"  Variance on Test Set: {var_test}")

        # Plot the results and save the plot as an image
        plt.figure()
        plt.scatter(X_train, y_train, color='blue', label='Training Data')
        X_plot = np.linspace(min(X_train), max(X_train), 100)
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, color='red', label=f'Fitted Curve (Degree {degree}, {reg_type})')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.title(f'Degree {degree} with {reg_type} Regularization')
        plt.savefig(f're_{i*3 + j + 1}.png')  # Save the plot as 're_1.png', 're_2.png', etc.
        plt.close()
