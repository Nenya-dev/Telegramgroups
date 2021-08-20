import nltk.cluster.util
from nltk.tokenize import WordPunctTokenizer
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from string import punctuation
import pprint as pp
import NLP
import pandas as pd
import numpy as np
from nltk.cluster import KMeansClusterer, euclidean_distance
from sklearn import cluster, metrics
from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import spacy
import matplotlib.pyplot as plt

# from wordcloud import WordCloud, ImageColorGenerator
posts = NLP.posts_prep_one
df = pd.DataFrame(list(posts.find()))
wpt = WordPunctTokenizer()
nlp = spacy.load("de_core_news_lg")


def vectorized():
    vectors = []
    for text in df['Message']:
        doc = nlp(text)
        for token in doc:
            vectors.append(token.vector)
            print('Vector for %s:' % token, token.vector)
        matrix_rows = []
        for vec in doc:
            row = [vec.similarity(token2) for token2 in doc]
            matrix_rows.append(row)

        similarity_matrix = np.array(matrix_rows)
        print(similarity_matrix)

    return vectors


def cluster_dbscan():
    vectors = vectorized()
    X, labels_true = make_blobs(n_samples=750, centers=vectors, cluster_std=0.4,
                               random_state=0)
    X = StandardScaler().fit_transform(X)

    dbscan = DBSCAN(metric='euclidean', eps=0.07, min_samples=3).fit(X)
    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    labels = dbscan.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.get_cmap('Spectral')(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def word2vec():
    print(df['Message'].head(10))
    messages = [wpt.tokenize(words) for words in df['Message']]
    '''
    for message in df.Message:
        if message not in punctuation and message not in stpwords:
            word_list.append(message)
'''
    print(messages[:10])
    print(len(messages))

    messages_vec = Word2Vec(messages,
                            min_count=5,
                            window=2,
                            sample=6e-5,
                            alpha=0.01,
                            min_alpha=0.0007
                            )

    model_name = "model_training"
    messages_vec.save(model_name)

    # Z = messages_vec.wv.
    # print(Z[0].shape)
    # print(Z[0])
    # Test
    # pp.pprint(messages_vec.wv.most_similar('merkel', topn=10))

    # centers, clusters = k_means(Z, 50)
    # centroid_map = dict(zip(messages_vec.wv.index2word, clusters))
    # top_words = get_top_words(messages_vec.wv.index2word, 20, centers, Z)

    return messages_vec


def k_means(word_vectors, num_cluster):
    model = word2vec()
    cluster_number = 10

    X = model[model.wv.vocab]
    kclusterer = KMeansClusterer(cluster_number, distance=nltk.cluster.util.cosine_distance, repeats=25)
    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        print(word + ":" + str(assigned_clusters[i]))

    kmeans = cluster.KMeans(n_clusters=num_cluster, init='k-means++')
    idx = kmeans.fit_predict(word_vectors)
    kmeans.fit(X)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return kmeans.cluster_centers_, idx


def get_top_words(index2word, k, centers, wordvecs):
    tree = KDTree
    # Closest points for each Cluster center is used to query the closest 20 points to it.
    closest_points = [tree.query(np.reshape(x, (1, -1)), k=k) for x in centers]
    closest_words_idxs = [x[1] for x in closest_points]

    # Word Index is queried for each position in the above array, and added to a Dictionary.
    closest_words = {}
    for i in range(0, len(closest_words_idxs)):
        closest_words['Cluster #' + str(i + 1).zfill(2)] = [index2word[j] for j in closest_words_idxs[i][0]]

    # A DataFrame is generated from the dictionary.
    df = pd.DataFrame(closest_words)
    df.index = df.index + 1

    return df


if __name__ == '__main__':
    vectorized()
    cluster_dbscan()
