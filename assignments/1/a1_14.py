import numpy as np
import matplotlib.pyplot as plt
import time
from collections import Counter

def euclidean_distance(X1, X2):
    return np.sqrt(np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2))

def manhattan_distance(X1, X2):
    return np.sum(np.abs(X1[:, np.newaxis] - X2), axis=2)

def cosine_distance(X1, X2):
    dot_product = np.dot(X1, X2.T)
    norm_X1 = np.linalg.norm(X1, axis=1, keepdims=True)
    norm_X2 = np.linalg.norm(X2, axis=1, keepdims=True)
    return 1 - (dot_product / (norm_X1 * norm_X2.T))

class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        if self.distance_metric == 'euclidean':
            distances = euclidean_distance(X, self.X_train)
        elif self.distance_metric == 'manhattan':
            distances = manhattan_distance(X, self.X_train)
        elif self.distance_metric == 'cosine':
            distances = cosine_distance(X, self.X_train)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        k_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = np.array([Counter(k).most_common(1)[0][0] for k in k_nearest_labels])
        return most_common

def measure_inference_time(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    start_time = time.time()
    model.predict(X_test)
    return time.time() - start_time

# Generate random dataset
np.random.seed(0)
sizes = [50, 100, 200, 300, 400, 500]
X = np.random.rand(max(sizes), 12)  # Max size of 500 samples
y = np.random.randint(0, 2, max(sizes))

# Split into train and test sets
def split_data(X, y, size):
    return X[:size], y[:size], X[size:], y[size:]

# Prepare data for plotting
results = {
    'size': [],
    'initial_knn': [],
    'best_knn': [],
    'optimized_knn': []
}

# Parameters for KNN models
initial_k = 3
best_k = 5  # Example value, use your best k from tuning
optimized_k = 7  # Example value, use your optimized k from tuning
best_metric = 'euclidean'  # Example value, use your best metric from tuning

for size in sizes:
    X_train, y_train, X_test, y_test = split_data(X, y, size)

    # Initialize models
    initial_knn = KNN(k=initial_k, distance_metric='euclidean')
    best_knn = KNN(k=best_k, distance_metric=best_metric)
    optimized_knn = KNN(k=optimized_k, distance_metric=best_metric)
    
    # Measure inference times
    initial_time = measure_inference_time(initial_knn, X_train, y_train, X_test)
    best_time = measure_inference_time(best_knn, X_train, y_train, X_test)
    optimized_time = measure_inference_time(optimized_knn, X_train, y_train, X_test)

    results['size'].append(size)
    results['initial_knn'].append(initial_time)
    results['best_knn'].append(best_time)
    results['optimized_knn'].append(optimized_time)

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(results['size'], results['initial_knn'], label='Initial KNN', marker='o')
plt.plot(results['size'], results['best_knn'], label='Best KNN', marker='o')
plt.plot(results['size'], results['optimized_knn'], label='Optimized KNN', marker='o')
plt.xlabel('Train Dataset Size')
plt.ylabel('Inference Time (seconds)')
plt.title('Inference Time vs Train Dataset Size for Custom KNN Models')
plt.legend()
plt.grid(True)
plt.savefig('inference_time_vs_size_custom_knn.png')
plt.show()

# Observations:
# 1. As the train dataset size increases, the inference time generally increases for all KNN models.
# 2. The initial KNN model may have higher inference times compared to optimized models due to its inefficiency.
# 3. The best KNN model might show better performance (lower inference time) compared to the initial model.
# 4. The optimized KNN model should ideally be faster than the initial model but may still vary depending on the optimizations applied.
