import numpy as np
from sklearn.decomposition import TruncatedSVD
import pandas as pd


class PC_Kmeans:
    def __init__(self, k, ml, cl, neighborhoods,columns, w = 0.1, tolerance=0.0001, max_iterations = 200):
        self.k = k 
        self.columns = columns
        self.ml = ml  # is transitive graph
        self.cl = cl  # is graph
        self.neighborhoods = neighborhoods
        self.w = w
        self.tolerance = tolerance
        self.max_iterations = max_iterations


    def fit(self, data):
        self.svd = TruncatedSVD(n_components=80)
        data_reduced = self.svd.fit_transform(data)
        self.centroids = {}
        cluster_centers = self.init_centers(data_reduced, self.neighborhoods)
        for i in range(len(cluster_centers)):
            self.centroids[i] = cluster_centers[i]

        for i in range(self.max_iterations):
            print("iteration:", i)
            self.clusters = {}
            for i in range(self.k):
                self.clusters[i] = set()

            alter = self.assign_clusters(data_reduced, cluster_centers, self.ml, self.cl, self.w)

            if alter == "empty":
                return "Cluster NOt Found"
            
            previous_centers = cluster_centers
            cluster_centers = self.mean_cluster_centers(data_reduced)
            isOptimal = True
            for i in range(len(cluster_centers)):
                original = previous_centers[i]
                curr = cluster_centers[i]
                diff = curr - original
                if np.sum(diff / original *100) > self.tolerance:
                    isOptimal= False

            if isOptimal:
                break
            
    def return_clusters(self):
        return self.clusters        
            


    def init_centers(self, data, neighborhoods):
        # neighborhoods = sorted(neighborhoods, key = len, reverse = True)
        neighborhood_centers = np.array([data[neighborhood].mean(axis=0) for neighborhood in neighborhoods])


        if len(neighborhoods) >= self.k:
            cluster_centers = neighborhood_centers[:self.k ]

        else:
            if len(neighborhoods) > 0:
                cluster_centers = neighborhood_centers
            else:
                cluster_centers = np.empty((0, data.shape[1]))

            if len(neighborhoods) < self.k:
                remaining_cluster_centers = data[np.random.choice(len(data), self.k - len(neighborhoods), replace=False), :]
                cluster_centers = np.concatenate([cluster_centers, remaining_cluster_centers])

        
        return cluster_centers

    def assign_clusters(self, data, cluster_centers, ml, cl, w):
        self.is_clustered = [-1] * len(data)     
        for x_index in range(len(data)):
            h = [self.loss_function(data, x_index, cluster_centers, cluster_index, self.is_clustered, 
                                         ml, cl, w) for cluster_index in range(self.k)]
            center_index = np.argmin(h)
            self.is_clustered[x_index] = center_index
            self.clusters[center_index].add(x_index)

        # Handle empty clusters
        empty_cluster_flag = False
        for i in range(self.k):
            if i not in self.is_clustered:
                empty_cluster_flag = True
                break
        if empty_cluster_flag:
            return "empty"


    def loss_function(self, data, x_index, centroids, cluster_index, is_clustered, ml, cl, w):
        distance = 1/ 2 * np.sum((data[x_index] - centroids[cluster_index]) **2)
        ml_penalty = 0
        for i in ml[x_index]:
            if is_clustered[i] != -1 and is_clustered[i] != cluster_index: 
                ml_penalty += w

        cl_penalty = 0
        for i in cl[x_index]: 
            if is_clustered[i] == cluster_index: 
                cl_penalty += w
        loss = distance  + ml_penalty + cl_penalty
        return loss

    

    def mean_cluster_centers(self, data): 
        for _center in self.clusters: 
            lst = []
            for index in self.clusters[_center]:
                lst.append(data[index])
            Arr = np.array(lst)

            if len(lst) != 0:
                self.centroids[_center] = np.mean(Arr, axis = 0)
        clusters_center = []
        for key, value in self.centroids.items():
            clusters_center.append(value)

        return np.array(clusters_center)
    

    def predict(self, data, centroids):
        labels = []
        for row in data:
            dist = []
            for centroid in centroids:
                distance = np.linalg.norm(row - centroid)
                dist.append(distance)
            classification = np.argmin(dist)
            labels.append(classification)

        return np.array(labels)


    def get_original_centroid_vector(self, data):
        centroids = self.centroids
        centroids_reduced_matrix = np.array([centroids[key] for key in sorted(centroids.keys())])
        centroids_original = np.dot(centroids_reduced_matrix, self.svd.components_)
        cluster_centers_df = pd.DataFrame(centroids_original, columns=data.columns)
        return cluster_centers_df