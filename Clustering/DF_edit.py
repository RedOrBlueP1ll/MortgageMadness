import pandas as pd

class DF_edit:

    """"Transpose a dataframe and append the new column to a new dataframe
        columns in article_tensor.csv - the article, the tensors"""
    def art_tens(self, df_to_transpose, df_to_add, save_to_location):
        df_transposed = df_to_transpose.T
        df_transposed.columns = ['article']
        #print(df_transposed)
        new_df = pd.concat([df_transposed, df_to_add], axis=1)
        new_df.to_csv(save_to_location, encoding='utf-8')

        return new_df

    """" Add a new column with the clusters to the dataframe and save it to cluster_article_tensor.csv 
            - has columns: cluster, article, the tensors"""
    def cl_art_tens(self, article_tensor_df, cluster_df, save_to_location):
        cluster_df.columns = ['cluster']
        new_df = pd.concat([cluster_df, article_tensor_df], axis=1)
        new_df.to_csv(save_to_location, encoding='utf-8')
        return new_df