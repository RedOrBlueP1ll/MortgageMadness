import time, os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# from umap import UMAP

class BGM:

        #method runs a cluster model depending using a form of BERT and clusters with Bayesian Gaussian Mixture
    def run_BGM(self, bert_embeddings_df, n_clusters):
        print("Loading embeddings...")

        # standardise embeddings for better reduction
        print('\nStandardising embeddings ...')
        start = time.time()
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(bert_embeddings_df)
        print('Duration: {} seconds'.format(round(time.time() - start, 3)))

        #reduce features
        # reduced_embeddings = UMAP(random_state=random_state, n_components=200).fit_transform(embeddings)

        print('\nClustering ...')
        start = time.time()
        BGM = BayesianGaussianMixture(n_components=n_clusters, random_state=1, n_init=1, covariance_type='full', init_params='kmeans', max_iter=100)
        BGM_model_labels = BGM.fit_predict(scaled_embeddings)
        print('Duration: {} seconds'.format(round(time.time() - start, 3)))

        labels_df = pd.DataFrame(BGM_model_labels)

        tensor_cluster_dict = {}
        j = 0
        for x in BGM_model_labels:
            tensor_cluster_dict[j] = x
            j += 1

        # return embeddings, model_labels
        return BGM_model_labels, tensor_cluster_dict, labels_df