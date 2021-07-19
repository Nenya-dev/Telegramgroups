# https://ai.intelligentonlinetools.com/ml/k-means-clustering-example-word2vec/
from gensim.models import Word2Vec
from sklearn import cluster
from sklearn import metrics
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

text_messages = tokenized

# Word2vec Modell
model=Word2Vec(text_messages, vector_size=100, workers=1)

words = list(model.wv.vocab)
print(words)

# Anzahl Cluser herausfinden mit Elbow Methode
X,y = vectors

plt.scatter(X[:,0], X[:1])
 wcss = []
 for i in range(1,11):
     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
     kmeans.fit(X)
     wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.show()

# Kmeans mit Word2vec

NUM_CLUSTER = 2
kmeans = cluster.KMeans(n_clusters=NUM_CLUSTER)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print("Cluster id labels for inputted data")
print(labels)
print("Centroids data")
print(centroids)

print(
    "Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print(kmeans.score(X))

silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

print("Silhouette_score: ")
print(silhouette_score)