from Kmeans import Kmeans
from DF_edit import DF_edit
from BGM import BGM
import pandas as pd

def main():
    # only change df_transpose and df_tensors
    # (note: the tensors with 383 dimensions have header while the others don't, hence different functions!)
    df_transpose = pd.read_csv('Training data/dimension_512/article_index_512.csv', header=None)
    df_tensors = pd.read_csv('Training data/dimension_512/sentence_tensors_512.csv', header=None)
    ##df_383 = pd.read_csv("Training data/tensorsBERT_383.csv")

    df_obj = DF_edit()
    save_to_location = 'Results Data/Kmeans/dimension_512/art_ten_dim512_clus50.csv'
    article_tensor_df = df_obj.art_tens(df_transpose, df_tensors, save_to_location) # create a new csv with the article and the tensor

    num_of_clusters = 50

    print("Performing KMeans...")
    kmeans_model = Kmeans()
    labels, cluster_dict, labels_df = kmeans_model.kmeans(df_tensors, num_of_clusters)
    save_to_location = 'Results Data/Kmeans/dimension_512/cl_art_ten_dim512_clus50.csv'
    cl_art_ten_df = df_obj.cl_art_tens(article_tensor_df, labels_df, save_to_location) # create a new csv with the cluster, article and the tensor
    full_df = pd.read_csv('Results Data/Kmeans/dimension_512/cl_art_ten_dim512_clus50.csv')

    pd.set_option('display.max_rows', None)
    print(full_df.groupby(['article', 'cluster']).sum()) # see how many clusters are in an article

    #print("Performing Elbow Method...")
    #kmeans_model.elbow_method(df_768)

    #print("Performing Silhouette Coefficients Method...")
    #kmeans_model.silCoeff(df_768)

def main2(dimension, num_of_clusters):
    # only change df_transpose and df_tensors
    # (note: the tensors with 383 dimensions have header while the others don't, hence different functions!)
    df_transpose = pd.read_csv('Training data/dimension_'+dimension+'/article_index_'+dimension+'.csv', header=None)
    df_tensors = pd.read_csv('Training data/dimension_'+dimension+'/sentence_tensors_'+dimension+'.csv', header=None)
    ##df_383 = pd.read_csv("Training data/tensorsBERT_383.csv")

    df_obj = DF_edit()
    save_to_location = 'Results Data/BGM/dimension_'+dimension+'/art_ten_dim'+dimension+'_clus'+str(num_of_clusters)+'.csv'
    article_tensor_df = df_obj.art_tens(df_transpose, df_tensors, save_to_location) #create a new csv with the article and the tensor

    print("Performing BGM...")
    BGM_model = BGM()
    labels, cluster_dict, labels_df = BGM_model.run_BGM(df_tensors, num_of_clusters)
    save_to_location = 'Results Data/BGM/dimension_' + dimension + '/cl_art_ten_dim' + dimension + '_clus' + str(num_of_clusters) + '.csv'
    cl_art_ten_df = df_obj.cl_art_tens(article_tensor_df, labels_df, save_to_location) #create a new csv with the cluster, article and the tensor

    # full_df = pd.read_csv(save_to_location)
    # pd.set_option('display.max_rows', None)
    # print(full_df.groupby(['article', 'cluster']).sum()) # see how many clusters are in an article

    #print("Performing Elbow Method...")
    #kmeans_model.elbow_method(df_768)

    #print("Performing Silhouette Coefficients Method...")
    #kmeans_model.silCoeff(df_768)


if __name__ == "__main__":
    # main()
    #this will do all of them at once
    dimensions = [32, 64, 128, 256, 384, 512]
    n_clusters = [10, 25, 50]
    for dim in dimensions:
        for n in n_clusters:
            print('WORKING ON', str(dim), '-', str(n))
            main2(dimension=str(dim), num_of_clusters=n)
    # main2(dimension=str(32), num_of_clusters=10)