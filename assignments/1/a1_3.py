import numpy as np
import matplotlib.pyplot as plt
import json
#Used LLM SERVICES for  fit and create polynomial function
class PolynomialRegression:
    def __init__(self, degree):
        self.degree = degree
        self.coefficients = None

    def fit(self, X, y):
        # Create polynomial features
        X_poly = self._create_polynomial_features(X)
        # Fit the model using the Normal Equation
        self.coefficients = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)

    def predict(self, X):
        # Create polynomial features
        X_poly = self._create_polynomial_features(X)
        return X_poly.dot(self.coefficients)

    def _create_polynomial_features(self, X):
        # Generate polynomial features (1, X, X^2, ..., X^degree)
        X_poly = np.ones((X.shape[0], 1))  # Column of ones for the intercept term
        for i in range(1, self.degree + 1):
            X_poly = np.c_[X_poly, X**i]
        return X_poly

    def save_model(self, filename):
        # Save the coefficients to a JSON file
        with open(filename, 'w') as file:
            json.dump(self.coefficients.tolist(), file)

    @staticmethod
    def load_model(filename):
        # Load the coefficients from a JSON file
        with open(filename, 'r') as file:
            coefficients = np.array(json.load(file))
        return coefficients

# Load the data from 'linreg.csv'
data = np.loadtxt('linreg.csv', delimiter=',', skiprows=1)
X = data[:, 0]  # Feature
y = data[:, 1]  # Target

# Shuffle the data
np.random.seed(0)
indices = np.random.permutation(len(X))
X, y = X[indices], y[indices]

# Split the data into train (80%) and test (20%) sets
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Test the model with degrees from 2 to 6
degrees = range(2, 7)
min_mse_test = float('inf')
best_k = None
best_model_filename = 'best_model.json'

for k in degrees:
    model = PolynomialRegression(degree=k)
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute MSE
    mse_train = np.mean((y_train - y_train_pred) ** 2)
    mse_test = np.mean((y_test - y_test_pred) ** 2)

    # Compute standard deviation and variance
    std_train = np.std(y_train_pred)
    var_train = np.var(y_train_pred)
    std_test = np.std(y_test_pred)
    var_test = np.var(y_test_pred)

    # Print metrics
    print(f"Degree {k}:")
    print(f"  MSE on Training Set: {mse_train}")
    print(f"  Standard Deviation on Training Set: {std_train}")
    print(f"  Variance on Training Set: {var_train}")
    print(f"  MSE on Test Set: {mse_test}")
    print(f"  Standard Deviation on Test Set: {std_test}")
    print(f"  Variance on Test Set: {var_test}")

    # Plot the training data and the fitted curve
    plt.figure()
    plt.scatter(X_train, y_train, color='blue', label='Training Data')
    X_plot = np.linspace(min(X_train), max(X_train), 100)
    y_plot = model.predict(X_plot)
    plt.plot(X_plot, y_plot, color='red', label=f'Fitted Curve (k={k})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Training Data with Fitted Polynomial Curve (Degree {k})')
    plt.savefig(f'lr3_{k - 1}.jpg')  # Save the image
    plt.show()

    # Update the best model if this one has the lowest test MSE
    if mse_test < min_mse_test:
        min_mse_test = mse_test
        best_k = k
        model.save_model(best_model_filename)

print(f"\nBest Degree: {best_k} with MSE on Test Set: {min_mse_test}")

# Load and display the best model's coefficients
best_coefficients = PolynomialRegression.load_model(best_model_filename)
print(f"Best Model Coefficients: {best_coefficients}")
