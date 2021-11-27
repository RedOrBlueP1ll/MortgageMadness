from kneed import KneeLocator
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

class Kmeans:

    """" Kmeans clustering algorithm - returns the labels of each tensor and returns a
            dictionary where the key is the n-th tensor and the value is the cluster that
            is assigned to the corresponding tensor """
    def kmeans(self, bert_embeddings_df,  num_clusters):
        print("Loading embeddings...")

        print("Standardizing embeddings...")
        # we create a scaler to stanadardize the embeddings
        # Standard Scaler - StandardScaler() will normalize the features i.e. each column of X, INDIVIDUALLY,
        # so that each column/feature/variable will have μ = 0 and σ = 1.
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(bert_embeddings_df)
        #print(scaled_features)

        print("Creating clusters...")
        kmeans_model_labels = KMeans(n_clusters=num_clusters, max_iter=100, random_state=43).fit_predict(scaled_embeddings)

        np.set_printoptions(threshold=np.inf) # this is to show all the clusters because it cuts them

        #print(kmeans_model_labels)
        labels_df = pd.DataFrame(kmeans_model_labels)
        #print(labels_df)

        tensor_cluster_dict = {}
        j = 0
        for x in kmeans_model_labels:
            tensor_cluster_dict[j] = x
            j += 1

        #print(tensor_cluster_dict)
        #print(kmeans_model_labels)



        return kmeans_model_labels, tensor_cluster_dict, labels_df

    """" The elbow method gives the optimal number of clusters using SSE - Sum of Squared Errors
            
        Elbow method gives us an idea on what a good k number of clusters would be based on the sum of 
        squared distance (SSE) between data points and their assigned clusters’ centroids. We pick k at 
        the spot where SSE starts to flatten out and forming an elbow."""
    def elbow_method(self, bert_embeddings_df):
        sse = []
        list_k = list(range(1, 51))
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(bert_embeddings_df)

        for k in list_k:
            km = KMeans(n_clusters=k)
            km.fit(scaled_embeddings)
            sse.append(km.inertia_)

        knee_locator = KneeLocator(range(1,51), sse, curve="convex", direction="decreasing")
        num_clusters = knee_locator.elbow
        print("Optimal number of clusters estimated from the Elbow Method: ", num_clusters)
        # Plot sse against k
        plt.figure(figsize=(6, 6))
        plt.plot(list_k, sse, '-o')
        plt.xlabel(r'Number of clusters *k*')
        plt.ylabel('Sum of squared distance')
        plt.show()

        return num_clusters

    """" The silhouette coefficient is a measure of cluster cohesion and separation. 
            It quantifies how well a data point fits into its assigned cluster based on two factors:

            1) How close the data point is to other points in the cluster
            2) How far away the data point is from points in other clusters
            
            Silhouette coefficient values range between -1 and 1. Larger numbers indicate that samples are 
            closer to their clusters than they are to other clusters.

            In the scikit-learn implementation of the silhouette coefficient, the average silhouette 
            coefficient of all the samples is summarized into one score. The silhouette score() function 
            needs a minimum of two clusters, or it will raise an exception.
            
            We want the coefficients to be as big as possible and close to 1 to have a good clusters"""
    def silCoeff(self, bert_embeddings_df):
        # a list that holds the silhouette couefficients
        sil_coeff = []

        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(bert_embeddings_df)

        for k in range(2, 51):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(scaled_embeddings)
            score = silhouette_score(scaled_embeddings, kmeans.labels_)
            sil_coeff.append(score)

        max_value_sil_coeff = max(sil_coeff)
        num_clusters = sil_coeff.index(max_value_sil_coeff) + 2

        print("Optimal number of clusters estimated from the Silhouette Coefficients: ", num_clusters)

        plt.plot(range(2,51), sil_coeff)
        plt.xticks(range(2,51))
        plt.xlabel("Number of clusters")
        plt.ylabel("Sil Coef")
        plt.show()

        return num_clusters