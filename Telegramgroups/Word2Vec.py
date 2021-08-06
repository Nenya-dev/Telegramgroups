# https://ai.intelligentonlinetools.com/ml/k-means-clustering-example-word2vec/
import nltk.cluster.util
from gensim.models import Word2Vec, word2vec
from sklearn import cluster
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from nltk.cluster import KMeansClusterer
import NLP
import spacy
import multiprocessing
import pandas as pd
from time import time
import numpy as np
import matplotlib as plt

nlp = spacy.load("de_core_news_lg")
posts = NLP.posts_prep_two
df = pd.DataFrame(list(posts.find()))
cores = multiprocessing.cpu_count()
wpt = nltk.WordPunctTokenizer


def word2vec_model():
    tokenized_words = [wpt.tokenize(words) for words in df.Message]

    # Word2vec model
    model = word2vec.Word2Vec(tokenized_words,
                              min_count=5,
                              window=2,
                              sample=6e-5,
                              alpha=0.01,
                              min_alpha=0.0007,
                              negative=20,
                              workers=cores - 1)

    word = model.wv.index2word
    wvs = model.wv[word]
    tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=2)
    np.set_printoptions(suppress=True)
    T = tsne.fit_transform(wvs)
    labels = word

    # Build Vocabulary Table
    t = time()
    sentences = NLP.detect_phrases()
    model_vocab = model.build_vocab(sentences, progress_per=10000)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    # Training model
    model.train(sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    # Test
    print(model.wv.most_similar(positive=['merkel']))
    print(model.wv.most_similar(positive=['querdenker']))

    return model_vocab


def avg_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros(num_features, dtype="float64")
    nwords = 0

    for word in words:
        if word in vocabulary:
            nwords = nwords + 1
            feature_vector = np.add(feature_vector, model[word])
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

        return feature_vector


def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    features = [avg_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                for tokenized_sentence in corpus]
    return np.array(features)


w2v_feature_array = averaged_word_vectorizer(corpus=, model=w2v_model,
                                             num_features=feature_size)
pd.DataFrame(w2v_feature_array)

def kmeans():
    model = word2vec()
    X = model[model.vocab]

    NUM_CLUSTERS = 5
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    print(assigned_clusters)

    words = list(model.vocab)
    for i, word in enumerate(words):
        print(word + ":" + str(assigned_clusters[i]))


if __name__ == '__main__':
    kmeans()
