from Kmeans import Kmeans
from DF_edit import DF_edit
import pandas as pd

def main():
    # only change df_transpose and df_tensors
    # (note: the tensors with 383 dimensions have header while the others don't, hence different functions!)
    df_transpose = pd.read_csv('Training data/article_index_128.csv', header=None)
    df_tensors = pd.read_csv('Training data/sentence_tensors_128.csv', header=None)
    df_383 = pd.read_csv("Training data/tensorsBERT_383.csv")

    df_obj = DF_edit()
    article_tensor_df = df_obj.art_tens(df_transpose, df_tensors) # create a new csv with the article and the tensor


    num_of_clusters = 10

    print("Performing KMeans...")
    kmeans_model = Kmeans()
    labels, cluster_dict, labels_df = kmeans_model.kmeans(df_tensors, num_of_clusters)
    cl_art_ten_df = df_obj.cl_art_tens(article_tensor_df, labels_df) # create a new csv with the cluster, article and the tensor
    full_df = pd.read_csv('Results Data/cluster_article_tensor.csv')

    pd.set_option('display.max_rows', None)
    #print(full_df.groupby(['article', 'cluster']).sum()) # see how many clusters are in an article

    #print("Performing Elbow Method...")
    #kmeans_model.elbow_method(df_768)

    #print("Performing Silhouette Coefficients Method...")
    #kmeans_model.silCoeff(df_768)

if __name__ == "__main__":
   main()