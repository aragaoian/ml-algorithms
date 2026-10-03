import numpy as np
import cv2

class KMeans:
    def __init__(self, n_clusters=3, init="random", n_init=10, max_iter=25, converge_tol=0.0001):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.converge_tol = converge_tol
        self.centroids = None
        self.labels = None

    def euclidean_distance(self, a, b):
        sum = 0
        for i in range(len(a)):
            sum += (a[i] - b[i])**2
        return np.sqrt(sum)

    def init_centroids(self, features):
        return np.array(
            [[np.random.randint(0, 256) for _ in range(features)] for _ in range(self.n_clusters)]
        )

    def calc_centroid(self, cluster):
        if len(cluster) == 0:
            return np.zeros_like(self.centroids[0])  # avoid nan if cluster is empty
        return np.mean(cluster, axis=0) # axis 0 to get the mean of each feature

    def fit(self, X):
        m, n = X.shape
        self.centroids = self.init_centroids(n)
        self.labels = np.zeros(m, dtype=int)

        for iter_counter in range(self.max_iter):
            clusters = [[] for _ in range(self.n_clusters)]

            # assign points to nearest centroid
            for i in range(m):
                point = X[i]
                distances = [self.euclidean_distance(point, c) for c in self.centroids]
                closest_index = np.argmin(distances)
                clusters[closest_index].append(point)
                self.labels[i] = closest_index

            new_centroids = np.array([self.calc_centroid(cluster) for cluster in clusters])
            
            shift = np.linalg.norm(self.centroids - new_centroids)
            if shift < self.converge_tol:
                break

            self.centroids = new_centroids
          
img1_copy = img1.copy()
img1_reshaped = img1_copy.reshape(-1, 3)

kmeans = KMeans()
kmeans.fit(X=img1_reshaped)

img_segmented = np.uint8(
    kmeans.centroids[kmeans.labels].reshape(img1_copy.shape)
)
