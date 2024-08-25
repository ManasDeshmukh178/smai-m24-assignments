import numpy as np
#UseD LLM services for Custom metric class 
class InitialKNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, Y):
        """Stores the training data."""
        self.X_train = X
        self.Y_train = Y

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
        k_nearest_labels = [self.Y_train[i] for i in k_nearest_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        """Predicts the labels for a set of instances."""
        return [self._predict(x) for x in X]

    def validate(self, X_val, Y_val):
        """Validates the model on validation data and calculates the metrics."""
        y_pred = self.predict(X_val)
        metrics = CustomMetrics()
        return metrics.validate(Y_val, y_pred)
