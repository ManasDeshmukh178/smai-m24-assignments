import pandas as pd
import numpy as np

# Define the KNN class
class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = self._compute_distances(x)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

    def _compute_distances(self, x):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        else:
            raise ValueError("Unsupported distance metric")

# Load the datasets
train_data = pd.read_csv('train.csv')
validate_data = pd.read_csv('validate.csv')
test_data = pd.read_csv('test.csv')

# Drop the first column (assuming it's an index or non-informative column)
train_data = train_data.drop(train_data.columns[0], axis=1)
validate_data = validate_data.drop(validate_data.columns[0], axis=1)
test_data = test_data.drop(test_data.columns[0], axis=1)

# Separate features and target variable
X_train = train_data.drop('target', axis=1).values
y_train = train_data['target'].values
X_validate = validate_data.drop('target', axis=1).values
y_validate = validate_data['target'].values
X_test = test_data.drop('target', axis=1).values
y_test = test_data['target'].values

# Initialize the KNN model with the best parameters
best_k = 5  # Replace with your best k
best_distance_metric = 'euclidean'  # Replace with your best distance metric
knn = KNN(k=best_k, distance_metric=best_distance_metric)

# Fit the model on the training data
knn.fit(X_train, y_train)

# Predict on the validation set to ensure accuracy is in the desired range
validate_predictions = knn.predict(X_validate)
validate_accuracy = np.mean(validate_predictions == y_validate)

# Print validation accuracy
print(f"Validation Accuracy: {validate_accuracy:.2f}")

# Predict on the test set
test_predictions = knn.predict(X_test)
test_accuracy = np.mean(test_predictions == y_test)

# Print test accuracy
print(f"Test Accuracy: {test_accuracy:.2f}")

# Observations
print("Observations:")
print("1. Accuracy on validation set:", validate_accuracy)
print("2. Accuracy on test set:", test_accuracy)
print("3. Ensure accuracy falls within the range of 0.25 to 0.32.")

# If the accuracy is not within the range, consider revisiting the hyperparameters or preprocessing steps.
