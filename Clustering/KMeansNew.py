from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

num_clusters = 51
def knn(num_clusters):
    print("Loading embeddings...")

    # input the tensors from BERT
    bert_embeddings_df = pd.read_csv("tensorsBERT.csv")

    print("Standardizing embeddings...")
    # we create a scaler to stanadardize the embeddings
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(bert_embeddings_df)
    #print(scaled_features)

    print("Creating clusters...")
    kmeans_model_labels = KMeans(n_clusters=num_clusters, max_iter=100, random_state=42).fit_predict(scaled_embeddings)
    np.set_printoptions(threshold=np.inf) # this is to show all the clusters because it cuts them

    j = 0
    for x in kmeans_model_labels:
        print("Tensor: ", j, " - Cluster: ", x)
        j += 1

    print(kmeans_model_labels)
    return kmeans_model_labels

    # -----------------------------------------------------------
    # the elbow method as evaluation
    # sse = []
    # list_k = list(range(1, 51))
    #
    # for k in list_k:
    #     km = KMeans(n_clusters=k)
    #     km.fit(bert_embeddings_df)
    #     sse.append(km.inertia_)
    #
    # # Plot sse against k
    # plt.figure(figsize=(6, 6))
    # plt.plot(list_k, sse, '-o')
    # plt.xlabel(r'Number of clusters *k*')
    # plt.ylabel('Sum of squared distance')
    # plt.show()


    return kmeans_model_labels

knn(num_clusters)