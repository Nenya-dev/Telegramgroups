import pandas as pd
from spacy.symbols import nsubj, VERB, ADJ, ADV
from textblob_de import TextBlobDE as tb
import nltk
from nltk.corpus import stopwords
import spacy
from Telegramgroups import NLP
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import numpy as np
from itertools import islice

posts = NLP.posts_prep_two
messages_original = NLP.posts_prep_one
nlp = spacy.load("de_core_news_sm")


def part_of_speech():
    pos = []
    name_entity = []

    df = pd.DataFrame(list(messages_original.find()))
    # df = text_data['Message'].convert_dtypes(convert_string=True)
    for message in df['Message']:
        nlp_message = nlp(message)
        # Name Entity Recognition
        for entity in nlp_message.ents:
            name_entity.append([entity.text, entity.label_])

        for token in nlp_message:
            pos.append([token.text, token.tag_, token.dep_])

        df['entity_text'] = pd.Series(name_entity)
        df['PoS'] = pd.Series(pos)
    df.to_pickle('PoS2.pkl')

    return df


def word_dependency():
    # df = pd.DataFrame(list(messages_original.find()))
    df = pd.read_pickle('PoS2.pkl')
   # print(df['PoS'].head(50))
    df1 = df.head(100)
    nlp.add_pipe("merge_entities")
    nlp.add_pipe("merge_noun_chunks")

    for doc in nlp.pipe(df.Message):
        for token in doc:
            print(token.dep_)
            if token.ent_type_ == "PER" or token.ent_type_ == "ORG":
                if token.dep_ in ('adc', 'avc'):
                    print("Token: ", token.text, "--->", "ADJ: ", token)
                else:
                    continue


def feature_extraction():
    df = pd.read_pickle('PoS.pkl')
    # for message in df['Message'].convert_dtypes(convert_string=True):
    cvec = CountVectorizer(min_df=.0025, max_df=.1, ngram_range=(1, 2))
    cvec.fit(df.Message)
    list(islice(cvec.vocabulary_.items(), 20))
    print(len(cvec.vocabulary_))

    # bag of words
    cvec_counts = cvec.transform(df.Message)
    # nonzero values
    print('sparse matrix shape:', cvec_counts.shape)
    print('nonzero count:', cvec_counts.nnz)
    print('sparsity: %.2f%%' % (100.0 * cvec_counts.nnz / (cvec_counts.shape[0] * cvec_counts.shape[1])))

    #  top 20 most common terms
    occ = np.asarray(cvec_counts.sum(axis=0)).ravel().tolist()
    counts_df = pd.DataFrame({'term': cvec.get_feature_names(), 'occurrences': occ})
    # print(counts_df.sort_values(by='occurrences', ascending=False).head(20))
    # weights
    transformer = TfidfTransformer()
    transformed_weights = transformer.fit_transform(cvec_counts)
    print(transformed_weights)
    weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
    print(weights_df.sort_values(by='weight', ascending=False).head(20))


def TFidf():
    df = pd.read_pickle('PoS.pkl')
    tvec = TfidfVectorizer(min_df=.0025, max_df=.1, ngram_range=(1, 2))
    tvec_weights = tvec.fit_transform(df.Message.dropna())
    weights = np.asarray(tvec_weights.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': tvec.get_feature_names(), 'weight': weights})
    print(weights_df.sort_values(by='weight', ascending=False).head(20))


if __name__ == '__main__':
    # part_of_speech()
    #   feature_extraction()
    # TFidf()
    word_dependency()
