import numpy as np
import matplotlib.pyplot as plt

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

# Add a column of ones to X to account for the intercept term (β0)
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Fit the model (Normal Equation: β = (X^T X)^-1 X^T y)
theta = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)

# Predictions
y_train_pred = X_train_b.dot(theta)
y_test_pred = X_test_b.dot(theta)

# Compute MSE
mse_train = np.mean((y_train - y_train_pred) ** 2)
mse_test = np.mean((y_test - y_test_pred) ** 2)

# Compute standard deviation and variance
std_train = np.std(y_train_pred)
var_train = np.var(y_train_pred)

std_test = np.std(y_test_pred)
var_test = np.var(y_test_pred)

# Print metrics
print(f"MSE on Training Set: {mse_train}")
print(f"Standard Deviation on Training Set: {std_train}")
print(f"Variance on Training Set: {var_train}")

print(f"MSE on Test Set: {mse_test}")
print(f"Standard Deviation on Test Set: {std_test}")
print(f"Variance on Test Set: {var_test}")

# Plot the training points and the fitted line
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_train, y_train_pred, color='red', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Training Data with Fitted Line')

# Save the plot as 'lin_reg2.jpg'
plt.savefig('lin_reg2.jpg')

# Optionally close the plot
plt.close()
