import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# Initial KNN class definition
class InitialKNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        """
        Initializes the KNN classifier with the number of neighbors (k)
        and the distance metric ('euclidean', 'manhattan', or 'cosine').
        """
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, Y):
        """
        Stores the training data. 
        This method memorizes the data as KNN is a lazy learner.
        """
        self.X_train = X
        self.Y_train = Y

    def calculate_distance(self, x1, x2):
        """
        Calculates distance between two points based on the selected distance metric.
        Supported metrics:
        - Euclidean: straight-line distance.
        - Manhattan: sum of absolute differences.
        - Cosine: 1 - cosine similarity.
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))  # Euclidean distance formula
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))  # Manhattan distance formula
        elif self.distance_metric == 'cosine':
            dot_product = np.dot(x1, x2)
            norm_x1 = np.linalg.norm(x1)
            norm_x2 = np.linalg.norm(x2)
            return 1 - (dot_product / (norm_x1 * norm_x2))  # Cosine similarity converted to distance

    def _predict(self, x):
        """
        Predicts the label for a single instance x.
        - Calculates distances from x to all training points.
        - Selects the k nearest neighbors.
        - Returns the most common label among the neighbors.
        """
        distances = [self.calculate_distance(x, x_train) for x_train in self.X_train]
        k_nearest_indices = np.argsort(distances)[:self.k]  # Indices of k smallest distances
        k_nearest_labels = [self.Y_train[i] for i in k_nearest_indices]  # Labels of k nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1)  # Most common label among neighbors
        return most_common[0][0]  # Return the label

    def predict(self, X):
        """
        Predicts the labels for a set of instances X.
        - Applies the _predict method for each instance.
        """
        return [self._predict(x) for x in X]

    def validate(self, X_val, Y_val):
        """
        Validates the model on validation data (X_val, Y_val).
        - Predicts labels for validation data.
        - Calculates and returns metrics using CustomMetrics.
        """
        y_pred = self.predict(X_val)
        metrics = CustomMetrics()
        return metrics.validate(Y_val, y_pred)

# CustomMetrics class definition
class CustomMetrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Calculates accuracy: the proportion of correct predictions.
        """
        correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
        return correct / len(y_true)

    @staticmethod
    def precision_recall_f1(y_true, y_pred):
        """
        Calculates precision, recall, and F1 scores.
        - Macro: Unweighted mean of precision/recall/F1 for each label.
        - Micro: Global precision/recall/F1 across all instances.
        """
        true_positive = Counter()  # Correctly predicted labels
        false_positive = Counter()  # Incorrectly predicted labels
        false_negative = Counter()  # Missed correct labels
        
        # Calculate true positives, false positives, and false negatives
        for y_t, y_p in zip(y_true, y_pred):
            if y_t == y_p:
                true_positive[y_t] += 1
            else:
                false_positive[y_p] += 1
                false_negative[y_t] += 1
        
        # Calculate precision, recall, and F1 for each label
        precision = {}
        recall = {}
        f1_score = {}
        
        for label in set(y_true):
            precision[label] = true_positive[label] / (true_positive[label] + false_positive[label]) if (true_positive[label] + false_positive[label]) > 0 else 0
            recall[label] = true_positive[label] / (true_positive[label] + false_negative[label]) if (true_positive[label] + false_negative[label]) > 0 else 0
            f1_score[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label]) if (precision[label] + recall[label]) > 0 else 0
        
        # Macro averages: average precision/recall/F1 across all labels
        precision_macro = np.mean(list(precision.values()))
        recall_macro = np.mean(list(recall.values()))
        f1_macro = np.mean(list(f1_score.values()))
        
        # Micro averages: global precision/recall/F1 across all instances
        precision_micro = sum(true_positive.values()) / (sum(true_positive.values()) + sum(false_positive.values())) if (sum(true_positive.values()) + sum(false_positive.values())) > 0 else 0
        recall_micro = sum(true_positive.values()) / (sum(true_positive.values()) + sum(false_negative.values())) if (sum(true_positive.values()) + sum(false_negative.values())) > 0 else 0
        f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0
        
        # Return all metrics as a tuple
        return precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro

    def validate(self, y_true, y_pred):
        """
        Calculates and returns a dictionary of evaluation metrics.
        - Accuracy
        - Precision, Recall, F1 (Macro)
        - Precision, Recall, F1 (Micro)
        """
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
