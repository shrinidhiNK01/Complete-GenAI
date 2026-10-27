"""
Decision Tree Classifier from Scratch (Gini Impurity, Splits) and Scikit-Learn.
This script demonstrates how decision boundaries are created step-by-step.

Run: python Complete_Course/ML/implementations/decision_tree.py
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Seed for reproducibility
np.random.seed(42)

class Node:
    """A helper node class for representing decision splits or leaf nodes."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature          # index of feature to split on
        self.threshold = threshold      # threshold value for split
        self.left = left                # left child node
        self.right = right              # right child node
        self.value = value              # prediction value if this is a leaf node

    def is_leaf_node(self):
        return self.value is not None


class ScratchDecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find the best split
        feat_indices = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_indices)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, feat_indices):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_indices:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                # Information gain (reduction in impurity)
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # Parent Gini
        parent_gini = self._gini(y)

        # Create split indices
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Weighted average Gini of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        gini_l, gini_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r

        # Information Gain is the reduction in impurity
        ig = parent_gini - child_gini
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _gini(self, y):
        # Gini Impurity = 1 - sum(p_i^2)
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)

    def _most_common_label(self, y):
        # Return the majority vote label
        if len(y) == 0:
            return None
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


def main():
    # 1. Generate standard binary moons dataset
    X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("--- Training Scratch Decision Tree ---")
    scratch_tree = ScratchDecisionTree(min_samples_split=5, max_depth=5)
    scratch_tree.fit(X_train, y_train)
    scratch_preds = scratch_tree.predict(X_test)

    print("--- Training Scikit-Learn Decision Tree ---")
    sklearn_tree = DecisionTreeClassifier(min_samples_split=5, max_depth=5, random_state=42)
    sklearn_tree.fit(X_train, y_train)
    sklearn_preds = sklearn_tree.predict(X_test)

    # Compare accuracies
    scratch_acc = accuracy_score(y_test, scratch_preds)
    sklearn_acc = accuracy_score(y_test, sklearn_preds)

    print("\n--- Model Performance Comparison ---")
    print(f"Scratch Accuracy:    {scratch_acc * 100:.2f}%")
    print(f"Sklearn Accuracy:    {sklearn_acc * 100:.2f}%")

if __name__ == '__main__':
    main()
