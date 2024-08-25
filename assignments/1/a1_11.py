import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product
from collections import Counter

# Define the InitialKNN class
class InitialKNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def calculate_distance(self, x1, x2):
        """Calculates distance between two points based on the selected metric."""
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
        """Predicts the label for a single instance x."""
        distances = [self.calculate_distance(x, x_train) for x_train in self.X_train]
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X_test):
        """Predicts the labels for a set of instances."""
        return [self._predict(x) for x in X_test]

    def validate(self, X_val, y_val):
        """Validates the model on validation data and calculates the accuracy."""
        y_pred = self.predict(X_val)
        return accuracy_score(y_val, y_pred)

# Load the dataset
df = pd.read_csv('dataset.csv')

# List of numerical features and target variable
numerical_features = ['popularity', 'duration_ms', 'danceability', 'energy', 
                      'key', 'loudness', 'speechiness', 'acousticness', 
                      'instrumentalness', 'liveness', 'valence', 'tempo']
target = 'track_genre'

# Prepare the features and target
X = df[numerical_features].values
y = df[target].values

# Split the data into train (80%), validation (10%), and test (10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize variables to store the best results
best_k = None
best_metric = None
best_accuracy = 0
results = []

# Define the range of k and the distance metrics to try
k_values = range(1, 31)
distance_metrics = ['euclidean', 'manhattan', 'cosine']

# Hyperparameter tuning
for k, distance_metric in product(k_values, distance_metrics):
    knn = InitialKNN(k=k, distance_metric=distance_metric)
    knn.fit(X_train, y_train)
    
    # Validate the model
    validation_accuracy = knn.validate(X_val, y_val)
    
    # Store results
    results.append((k, distance_metric, validation_accuracy))
    
    # Update the best model
    if validation_accuracy > best_accuracy:
        best_k = k
        best_metric = distance_metric
        best_accuracy = validation_accuracy

# Sort the results by validation accuracy in descending order
results.sort(key=lambda x: x[2], reverse=True)

# Print the top 10 results
print("Top 10 {k, distance metric} pairs by validation accuracy:")
for i, (k, metric, accuracy) in enumerate(results[:10]):
    print(f"{i + 1}. k={k}, distance_metric='{metric}', accuracy={accuracy:.4f}")

# Plot k vs accuracy for the best distance metric
best_metric_results = [result for result in results if result[1] == best_metric]
k_values_best_metric = [result[0] for result in best_metric_results]
accuracies_best_metric = [result[2] for result in best_metric_results]

plt.figure(figsize=(10, 6))
plt.plot(k_values_best_metric, accuracies_best_metric, marker='o')
plt.title(f'Accuracy vs. k for distance metric: {best_metric}')
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.show()

# Print the best {k, distance metric} pair and its accuracy
print(f"\nBest {k, distance_metric} pair:")
print(f"Best k: {best_k}")
print(f"Best distance metric: {best_metric}")
print(f"Best validation accuracy: {best_accuracy:.4f}")
