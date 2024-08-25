import numpy as np
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
#Used LLM services  for polynomial regression
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

# Function to create frames for visualization
def create_frames(X_train, y_train, X_test, y_test, degrees):
    frame_paths = []  # Store paths of saved frames

    for k in degrees:  # Loop through the specified degrees
        model = PolynomialRegression(degree=k)  # Initialize model for current degree
        model.fit(X_train, y_train)  # Fit model to training data

        y_train_pred = model.predict(X_train)  # Predict on training data
        y_test_pred = model.predict(X_test)  # Predict on test data

        # Calculate MSE, standard deviation, and variance for training and test sets
        mse_train = np.mean((y_train - y_train_pred) ** 2)
        mse_test = np.mean((y_test - y_test_pred) ** 2)
        std_train = np.std(y_train_pred)
        var_train = np.var(y_train_pred)
        std_test = np.std(y_test_pred)
        var_test = np.var(y_test_pred)

        # Print metrics for the current degree
        print(f"\nDegree {k}:")
        print(f"  MSE on Training Set: {mse_train}")
        print(f"  Standard Deviation on Training Set: {std_train}")
        print(f"  Variance on Training Set: {var_train}")
        print(f"  MSE on Test Set: {mse_test}")
        print(f"  Standard Deviation on Test Set: {std_test}")
        print(f"  Variance on Test Set: {var_test}")

        plt.figure(figsize=(10, 8))

        # Plot polynomial regression curve
        plt.subplot(2, 2, 1)
        plt.scatter(X_train, y_train, color='blue', label='Training Data')
        X_plot = np.linspace(min(X_train), max(X_train), 100)  # Generate X values for plotting the curve
        y_plot = model.predict(X_plot)  # Predict y values for plotting the curve
        plt.plot(X_plot, y_plot, color='red', label=f'Fitted Curve (k={k})')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Polynomial Regression (Degree {k})')

        # Plot Mean Squared Error (MSE)
        plt.subplot(2, 2, 2)
        plt.bar(['Train', 'Test'], [mse_train, mse_test], color=['blue', 'red'])
        plt.title('Mean Squared Error')
        plt.ylabel('MSE')

        # Plot Standard Deviation
        plt.subplot(2, 2, 3)
        plt.bar(['Train', 'Test'], [std_train, std_test], color=['blue', 'red'])
        plt.title('Standard Deviation')
        plt.ylabel('Standard Deviation')

        # Plot Variance
        plt.subplot(2, 2, 4)
        plt.bar(['Train', 'Test'], [var_train, var_test], color=['blue', 'red'])
        plt.title('Variance')
        plt.ylabel('Variance')

        plt.tight_layout()  # Adjust layout to avoid overlap

        frame_path = f'reg_{k}.png'  # Save the plot as an image
        plt.savefig(frame_path)
        plt.close()  # Close the plot to avoid displaying it

        frame_paths.append(frame_path)  # Add frame path to list

    return frame_paths  # Return the list of frame paths

# Function to create a GIF from the saved frames
def create_gif(frame_paths, gif_path, duration=700):
    frames = [Image.open(frame) for frame in frame_paths]  # Load frames from paths
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)  # Save as GIF

# Load dataset
data = np.loadtxt('linreg.csv', delimiter=',', skiprows=1)
X = data[:, 0]
y = data[:, 1]

# Shuffle data randomly
np.random.seed(0)
indices = np.random.permutation(len(X))
X, y = X[indices], y[indices]

# Split data into training (80%) and test (20%) sets
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Specify polynomial degrees to fit
degrees = range(2, 7)

# Generate frames and save paths
frame_paths = create_frames(X_train, y_train, X_test, y_test, degrees)

# Create GIF from frames
gif_path = 'convergence.gif'
create_gif(frame_paths, gif_path)

# Print the path where the GIF is saved
print(f"\nGIF saved at: {gif_path}")
