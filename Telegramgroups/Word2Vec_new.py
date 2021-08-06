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
# from wordcloud import WordCloud, ImageColorGenerator
posts = NLP.posts_prep_one
df = pd.DataFrame(list(posts.find()))
wpt = WordPunctTokenizer()


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

    Z = messages_vec.wv.
    print(Z[0].shape)
    print(Z[0])
    # Test
    # pp.pprint(messages_vec.wv.most_similar('merkel', topn=10))

    centers, clusters = k_means(Z, 50)
    centroid_map = dict(zip(messages_vec.wv.index2word, clusters))
    top_words = get_top_words(messages_vec.wv.index2word, 20, centers, Z)

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
    word2vec()
