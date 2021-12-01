import time, os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import MinMaxScaler
# from umap import UMAP

    #method runs a cluster model depending using a form of BERT and clusters with Bayesian Gaussian Mixture
def run_bert_BGM_model(n_clusters, random_state):

    #here we create the claim embeddings (document vectors) with BERT (767 dimensions per text)
    embeddings = get_BERT_embeddings()

    # standardise embeddings for better reduction
    print('\nStandardising embeddings ...')
    start = time.time()
    # embeddings = StandardScaler().fit_transform(embeddings)
    embeddings = MinMaxScaler().fit_transform(embeddings)
    print('Duration: {} seconds'.format(round(time.time() - start, 3)))

    #reduce features
    # reduced_embeddings = UMAP(random_state=random_state, n_components=200).fit_transform(embeddings)

    print('\nClustering ...')
    start = time.time()
    model_labels = BayesianGaussianMixture(random_state=random_state, n_components=n_clusters, n_init=4, weight_concentration_prior=0.05, covariance_type='full', init_params='kmeans', max_iter=300).fit_predict(embeddings)
    print('Duration: {} seconds'.format(round(time.time() - start, 3)))

    return embeddings, model_labels

    #here we get the BERT embeddings from the BERT class
def get_BERT_embeddings():
    embeddings = pd.read_csv("/home/roobs/Documents/PycharmProjects/MortgageMadness/Clustering/Results Data/cluster_article_tensor.csv")
    return embeddings

    #testing
if __name__ == '__main__':
    #some testing
    print('Somtheing')
    embeddings, model_labels = run_bert_BGM_model(n_clusters=30, random_state=0)
    print(model_labels)