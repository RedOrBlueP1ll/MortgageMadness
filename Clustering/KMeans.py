# credit: https://realpython.com/k-means-clustering-python/#what-is-clustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# this is for testing
# input the pdf instead of the random text
document = 'This is the most beautiful place in the world . This man has more skills to show in cricket than any other game., “Hi there! how was your ladakh trip last month?”, “There was a player who had scored 200+ runs in single cricket innings in his career.”, “I have got the opportunity to travel to Paris next year for my internship.”, “May be he is better than you in batting but you are much better than him in bowling.”, “That was really a great day for me when I was there at Lavasa for the whole night.”, “That’s exactly I wanted to become, a highest ratting batsmen ever with top scores.”, “Does it really matter wether you go to Thailand or Goa, its just you have spend your holidays.”, “Why don’t you go to Switzerland next year for your 25th Wedding anniversary?”, “Travel is fatal to prejudice, bigotry, and narrow mindedness., and many of our people need it sorely on these accounts.”, “Stop worrying about the potholes in the road and enjoy the journey.”, “No cricket team in the world depends on one or two players. The team always plays to win.”, “Cricket is a team game. If you want fame for yourself, go play an individual game.”, “Because in the end, you won’t remember the time you spent working in the office or mowing your lawn. Climb that goddamn mountain.”, “Isn’t cricket supposed to be a team sport? I feel people should decide first whether cricket is a team game or an individual sport.'
document = document.split('.')
question = ['Nothing is easy in cricket. Maybe when you watch it on TV, it looks easy. But it is not. You have to use your brain and time the ball.']



kmeans_kwargs = {"n_init": 10, "max_iter": 300}


#predict the number of clusters with the elbow method
#the elbow method uses SSE (sum of the squared error)
def elbowMethod():
    sse = [] # a list to hold the SSE value for each k
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(document)
    for k in range(15,51):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X)
        sse.append(kmeans.inertia_) #kmeans.inertia_ => the lowest SSE value

    plt.style.use("fivethirtyeight")
    plt.plot(range(1,11), sse)
    plt.xticks(range(1,11))
    plt.xlabel("Num of Clusters")
    plt.ylabel("SSE")
    plt.show()

    kl = KneeLocator(range(1,11), sse, curve="convex", direction="decreasing")
    clusters = kl.elbow # this will be the number of clusters put out from the elbow method
    print('clusters with elbow:', clusters)
    return clusters

#predict the number of clusters with the silhouette coefficient
#the silhouette coefficient quantifies how well a data point fits into its assigned cluster based on two factors:
# a) how close the data point is to other points in the cluster
# b) how far away the data point is from points in other clusters
# silhouette coefficient values are between -1 and 1.
# larger numbers indicate that samples are closer to their clusters than they are to other clusters
def silCoeff():
    silhouette_coefficients = []
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(document)

    for k in range(2,11): #the range starts at 2 because the silhouette score function needs a minimum of 2 clusters or it gives an exception
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_coefficients.append(score)

    plt.style.use("fivethirtyeight")
    plt.plot(range(2,11), silhouette_coefficients)
    plt.xticks(range(2,11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sil Coeff")
    plt.show()


def kmeansClustering(document, question):
    # change this to BERT vectors
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(document)
    #print(X)

    # use the elbowMethod() or silhouetteCoefficient() to determine the best number of clusters
    #k = elbowMethod()
    #k = silCoeff()
    k = 5  # for now
    print("Number of clusters: ", k)
    # init - controls the initialization technique
    # n_clusters - number of clusters
    # n_init - sets the number of initializations to perform (default is to perform 10 k-means runs and return the results of the one with the lowest SSE
    # max_iter - sets the number of max iterations for each initialization of the k-means
    model = KMeans(n_clusters=k, max_iter=100)
    model.fit(X)

    # TEST
    print('\n')
    print('Prediction')
    # input the question
    question = vectorizer.transform(question)
    predicted = model.predict(question)
    print('Cluster:', predicted)

    return k, predicted #k - the number of clusters, predicted - the predicted cluster

#test
#kmeansClustering(document, question)