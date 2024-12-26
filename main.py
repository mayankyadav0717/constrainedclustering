#=========================  Importing required libraries ==============================

from process import PreProcess
import pandas as pd 
import matplotlib.pyplot as plt
import argparse
from bag_of_words import BagOfWords
from candidate_selection import  select_candidates
from process import PreProcess
from constraints_generation import generate_constraints, transitive_entailment_graph, remove_zero_val_words
from PCKmeans import PC_Kmeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from select_neighborhood import select_top_neighborhoods_with_cannot_link
import spacy
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import fetch_20newsgroups
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,davies_bouldin_score, adjusted_rand_score
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from word_embeddings import get_document_vector
nlp = spacy.load("en_core_web_md")

#=======================================================================================

# Initialize ArgumentParser
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--k", type=int, required=True, help="An integer value for k")
parser.add_argument("--query", type=int, required=True, help="An integer value for data point")

try:
    args = parser.parse_args()
    k = args.k
    row_number = args.query  # Assuming 'query' is the row index of the dataframe
    print(f"Received arguments: k = {k}, query = {row_number}")

except argparse.ArgumentError as e:
    print(f"Error parsing arguments: {e}")
except Exception as e:
    print(f"An error occurred: {e}")



# Defining the categories we want to fetch

categories = [
    'talk.religion.misc',
    'comp.graphics',
    'sci.med',
    'rec.autos'
]

#======================= Fetching the 20NewsGroupData from API =========================

newsgroups = fetch_20newsgroups(subset='train', categories= categories, remove = ('headers', 'footers', 'quotes'))
df = pd.DataFrame({'doc': newsgroups.data, 'category': newsgroups.target})
indices_to_drop = []
for i in range(len(df)):
    if len(df['doc'][i]) == 0:
        indices_to_drop.append(i)
df.drop(index=indices_to_drop, inplace=True)
df.reset_index(drop=True, inplace=True)
process = PreProcess(df)
df = process.process()
tfidf = BagOfWords(df, 'doc')
tf_idf_matrix, tf_idf_df = tfidf.tf_idf()

#=================== Selection of the data points for clustering =======================

if  row_number >=0 and len(df) >= row_number:
    tf_idf_query_vector = tf_idf_df.loc[int(row_number)]
    closest = select_candidates(k, tf_idf_df, tf_idf_query_vector)
    selected_datapoints = df.iloc[closest, :].reset_index(drop=True)
else:
    print(f"Row number {row_number} is out of bounds for the DataFrame.")

#=======================================================================================

#===== Re-evaluating the TFIDF matrix based on the selected datapoints =================

indices_to_drop = []
for i in range(len(selected_datapoints)):
    if len(selected_datapoints['doc'][i]) == 0:
        indices_to_drop.append(i)
selected_datapoints.drop(index=indices_to_drop, inplace=True)
selected_datapoints.reset_index(drop=True, inplace=True)
selected_datapoints_tfidf = BagOfWords(selected_datapoints, 'doc')
selected_datapoints_matrix, selected_datapoints_tfidf_df = selected_datapoints_tfidf.tf_idf()
column_names = selected_datapoints_tfidf_df.columns.tolist()

#=======================================================================================

#==================  Keyphrase Extraction Using Rake  ==================================
def extract_key_phrases(texts):
    """Extract key phrases using SpaCy."""
    key_phrases = []
    for doc in nlp.pipe(texts):
        phrases = [chunk.text for chunk in doc.noun_chunks]
        key_phrases.append(" ".join(phrases))  # Join phrases into a single string
    return key_phrases

#=======================================================================================

#======================= Generating Must Link/ Cannot Link Constraint  =================
k = 100
if selected_datapoints_tfidf_df.shape[1] >= k:
    svd = TruncatedSVD(n_components=80) #reduced number dimensions 
    _matrix_ = svd.fit_transform(selected_datapoints_matrix)
else: 
    _matrix_ = selected_datapoints_matrix.toarray()
similarity_matrix = cosine_similarity(_matrix_)
selected_ds_len = len(selected_datapoints)
document_keyphrases_list = []
for i in range(len(selected_datapoints_tfidf_df)):
    words = remove_zero_val_words(selected_datapoints_tfidf_df, i)
    document_keyphrases_list.append(set(words))

must_link_constraints, cannot_link_constraints = generate_constraints(document_keyphrases_list, selected_datapoints_tfidf_df, similarity_matrix)
#========================================================================================

#=====Now we Generate the Neighbours based on must link and cannot link constraints=====

neighborhoods, ml, cl = transitive_entailment_graph(must_link_constraints, cannot_link_constraints, selected_ds_len)
best_neighborhoods = select_top_neighborhoods_with_cannot_link(neighborhoods, cannot_link_constraints)

#=======================================================================================

#========  PCKMeans algorithm implementation for cluster generation =====================
# We can choose the feature fr clustering
# selected_feature = ["tfidf", 'key_phrases','word_embeddings' ]
selected_feature = ['tfidf']
for selected_feature in selected_feature:
    if selected_feature == 'tfidf':
        pck = PC_Kmeans(4, ml, cl, best_neighborhoods, column_names)
        clusters = pck.fit(selected_datapoints_matrix )
        cluster_assignments_list = []
        cluster_assignments = pck.clusters
        for cluster, indices in cluster_assignments.items():
            for index in indices:
                cluster_assignments_list.append((index, cluster))
        clusters_centers_df = pck.get_original_centroid_vector(selected_datapoints_tfidf_df)

    if selected_feature == 'word_embeddings':
        num_dimensions = 300
        feature_names = [f'word_{i+1}' for i in range(num_dimensions)]
        vector_list = []
        for document in selected_datapoints["doc"]: 
            vector_list.append(get_document_vector(nlp, document))
        vector_array = np.array(vector_list)
        embedding_df = pd.DataFrame(vector_list, columns=feature_names)
        pck = PC_Kmeans(4, ml, cl, best_neighborhoods, column_names)
        clusters = pck.fit(vector_array )
        cluster_assignments_list = []
        cluster_assignments = pck.clusters
        for cluster, indices in cluster_assignments.items():
            for index in indices:
                cluster_assignments_list.append((index, cluster))
        clusters_centers_df = pck.get_original_centroid_vector(embedding_df)

    if selected_feature == 'key_phrases':
        selected_datapoints['key_phrases'] = extract_key_phrases(selected_datapoints['doc'])
        vectorizer_key_phrases = TfidfVectorizer(stop_words='english')
        tfidf_matrix_keyphrases = vectorizer_key_phrases.fit_transform(selected_datapoints['key_phrases'])
        tfidf_df_keyphrases = pd.DataFrame(tfidf_matrix_keyphrases.toarray(), columns=vectorizer_key_phrases.get_feature_names_out())
        pck = PC_Kmeans(4, ml, cl, best_neighborhoods, column_names)
        clusters = pck.fit(tfidf_matrix_keyphrases)
        cluster_assignments_list = []
        cluster_assignments = pck.clusters
        for cluster, indices in cluster_assignments.items():
            for index in indices:
                cluster_assignments_list.append((index, cluster))
        clusters_centers_df = pck.get_original_centroid_vector(tfidf_df_keyphrases)

    
    # Convert the list to a DataFrame
    cluster_df = pd.DataFrame(cluster_assignments_list, columns=['index', 'cluster'])
    
    #=======================================================================================

    cluster_df = cluster_df.sort_values('index').reset_index(drop=True)
    selected_datapoints['cluster'] = cluster_df['cluster']
    selected_datapoints['cluster'].astype(str)

    #=================== NaiveBayesClassifier, ChiSq Feature Selection  ======================

    no_of_features = 100 # Number of features to select
    chi2_selector = SelectKBest(chi2, k=no_of_features)
    X_chi2_selected = chi2_selector.fit_transform(selected_datapoints_tfidf_df, selected_datapoints['cluster'])

    #We are using a Naive Bayes Classifier to use Lime TEXT Explaination
    classifier = MultinomialNB()
    classifier.fit(X_chi2_selected, selected_datapoints['cluster'])
    y_pred = classifier.predict(X_chi2_selected)
    print(f"Accuracy Score for Naive Bayes Classifier fitted on Clustered Data \n: {accuracy_score(selected_datapoints['cluster'], y_pred)}")

    #========================================================================================

    #============================= Lime Text Explainer ======================================

    sample_index = 20
    sample_text = selected_datapoints['doc'][sample_index] # Sample text data
    LTPipeline = make_pipeline(selected_datapoints_tfidf.vectorizer, chi2_selector, classifier) #Creating LIME Text Explainer
    clusters = selected_datapoints['cluster'].unique()
    explainer = LimeTextExplainer(class_names=clusters)
    exp = explainer.explain_instance(sample_text, LTPipeline.predict_proba, num_features=10, top_labels=1) # Function to transform text to features
    exp.save_to_file('lime_explanation.html')

    #=======================================================================================

    #=================   Creating Rule Based Explanations    ===============================  
 
    top_features_per_cluster = []
    for cluster_idx in range(clusters_centers_df.shape[0]):
        centroid = clusters_centers_df.iloc[cluster_idx]
        top_features_idx = centroid.argsort()[-5:][::-1]
        top_features = clusters_centers_df.columns[top_features_idx]
        top_features_per_cluster.append(top_features)

    for cluster_idx, top_features in enumerate(top_features_per_cluster):
        print(f"Cluster {cluster_idx}: {', '.join(top_features)}")

    #========================================================================================
      
    #=================  Calculating various internal metrics  ===============================
    silhouette_avg = silhouette_score(selected_datapoints_tfidf_df, selected_datapoints['cluster'])
    davies_bouldin = davies_bouldin_score(selected_datapoints_tfidf_df, selected_datapoints['cluster'])
    adjusted_rand_score_ = adjusted_rand_score(selected_datapoints['category'], selected_datapoints['cluster'])
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")
    print(f"adjusted_rand_score_: {adjusted_rand_score_}")

    #=======================================================================================

    #================  Plotting the clusters with intial Centroids    ======================

    neighborhoods1 = sorted(best_neighborhoods, key = len, reverse = True)[:4]
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(selected_datapoints_tfidf_df)
    pca_centroids = np.array([pca_result[index].mean(axis = 0) for index in neighborhoods1])
    labels = selected_datapoints["cluster"]
    label1 = selected_datapoints['cluster'].unique()
    umap_ = umap.UMAP(n_components=2, random_state=2)
    umap_result = umap_.fit_transform(selected_datapoints_tfidf_df.values)
    umap_centroids = np.array([umap_result[index].mean(axis = 0) for index in neighborhoods1])
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', alpha=0.3)
    axs[0].set_title('PCA')
    axs[2].scatter(umap_result[:, 0], umap_result[:, 1], c=labels, cmap='viridis', alpha = 0.3)
    axs[2].set_title('UMAP')
    axs[0].scatter(pca_centroids[:, 0], pca_centroids[:, 1], c=label1, marker='*', s=200)
    axs[2].scatter(umap_centroids[:, 0], umap_centroids[:, 1],c=label1, marker='*', s=200)
    plt.show()


#=================================  Thank You   =========================================