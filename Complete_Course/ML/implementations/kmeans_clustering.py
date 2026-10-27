"""
K-Means Clustering from Scratch (Distance, Centroid Updates) and Scikit-Learn.
This script demonstrates unsupervised clustering.

Run: python Complete_Course/ML/implementations/kmeans_clustering.py
"""

import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Set seed
np.random.seed(42)

class ScratchKMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = []
        self.classes = {}

    def _euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # 1. Randomly initialize centroids from existing data points
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = [X[idx] for idx in random_indices]

        # 2. Optimization loop
        for i in range(self.max_iters):
            # Create empty classifications dictionary {0: [], 1: [], ...}
            self.classes = {j: [] for j in range(self.k)}
            
            # 3. Assign each sample to the nearest centroid
            for sample in X:
                distances = [self._euclidean_distance(sample, centroid) for centroid in self.centroids]
                classification = np.argmin(distances)
                self.classes[classification].append(sample)

            previous_centroids = list(self.centroids)

            # 4. Update centroid positions by taking the mean of all points in that cluster
            for classification in self.classes:
                # If a cluster is empty, don't update (keep original centroid)
                if len(self.classes[classification]) > 0:
                    self.centroids[classification] = np.mean(self.classes[classification], axis=0)

            # 5. Check convergence (if centroids did not change, stop early)
            is_converged = True
            for centroid_idx in range(self.k):
                prev = previous_centroids[centroid_idx]
                curr = self.centroids[centroid_idx]
                if np.sum((curr - prev) / (prev + 1e-9)) > 0.0001:
                    is_converged = False
                    break
            
            if is_converged:
                print(f"K-Means converged at iteration {i}")
                break

    def predict(self, X):
        predictions = []
        for sample in X:
            distances = [self._euclidean_distance(sample, centroid) for centroid in self.centroids]
            predictions.append(np.argmin(distances))
        return np.array(predictions)


def main():
    # 1. Generate multi-cluster blob dataset
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

    # 2. Run Scratch K-Means
    print("--- Training Scratch K-Means ---")
    scratch_kmeans = ScratchKMeans(k=4, max_iters=100)
    scratch_kmeans.fit(X)
    scratch_labels = scratch_kmeans.predict(X)

    # 3. Run Sklearn K-Means
    print("\n--- Training Scikit-Learn K-Means ---")
    sklearn_kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    sklearn_kmeans.fit(X)
    sklearn_labels = sklearn_kmeans.predict(X)

    # Convert centroids to numpy array for plotting
    scratch_centroids = np.array(scratch_kmeans.centroids)
    sklearn_centroids = sklearn_kmeans.cluster_centers_

    # Plot Comparison
    plt.figure(figsize=(12, 5))

    # Plot 1: Scratch Model
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=scratch_labels, cmap='viridis', alpha=0.7, edgecolors='k')
    plt.scatter(scratch_centroids[:, 0], scratch_centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
    plt.title("Scratch K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    # Plot 2: Sklearn Model
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=sklearn_labels, cmap='plasma', alpha=0.7, edgecolors='k')
    plt.scatter(sklearn_centroids[:, 0], sklearn_centroids[:, 1], c='cyan', s=200, marker='o', edgecolors='k', label='Centroids')
    plt.title("Sklearn K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    plt.tight_layout()
    plt.savefig("Complete_Course/ML/kmeans_clustering_results.png")
    print("\nPlot saved successfully as 'Complete_Course/ML/kmeans_clustering_results.png'")

if __name__ == '__main__':
    main()
