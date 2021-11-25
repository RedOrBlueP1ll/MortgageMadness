from Kmeans import Kmeans
import pandas as pd

def main():
    num_of_clusters = 51
    df = pd.read_csv("tensorsBERT.csv")

    print("Performing KMeans...")
    kmeans_model = Kmeans()
    kmeans_model.kmeans(df, num_of_clusters)

    print("Performing Elbow Method...")
    kmeans_model.elbow_method(df)

    print("Performing Silhouette Coefficients Method...")
    kmeans_model.silCoeff(df)

if __name__ == "__main__":
   main()