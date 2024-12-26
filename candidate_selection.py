import numpy as np
from scipy.spatial.distance import cdist

def dist(doc_features, query_features):
    if hasattr(doc_features, 'toarray'):
        doc_features = doc_features.toarray()
    if hasattr(query_features, 'toarray'):
        query_features = query_features.toarray().reshape(1, -1)
    else:
        query_features = np.array(query_features).reshape(1, -1) 
    dists = cdist(doc_features, query_features, metric='euclidean').flatten()
    return dists

def select_candidates(k, doc_features, query_features):
    distances = dist(doc_features, query_features)
    non_zero_indices = np.where(distances > 0)[0]  
    non_zero_distances = distances[non_zero_indices]
    sorted_non_zero_indices = non_zero_indices[np.argsort(non_zero_distances)]
    top_k_indices = sorted_non_zero_indices[:k]
    return top_k_indices




