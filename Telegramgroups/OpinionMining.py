# https://medium.com/analytics-vidhya/aspect-based-sentiment-analysis-a-practical-approach-8f51029bbc4a
import pickle
import pprint
import pandas as pd
import stanfordnlp
import nltk
import os
from nltk.corpus import stopwords
import spacy
from Telegramgroups import NLP
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import numpy as np
from itertools import islice
from nltk.tag.stanford import StanfordNERTagger
import stanza
import spacy_stanza
from stop_words import get_stop_words
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob_de import TextBlobDE
from nltk import word_tokenize
MODELS_DIR = '.'
# stanfordnlp.download('de', MODELS_DIR)
posts = NLP.posts_prep_two
messages_original = NLP.posts_prep_one
nlp = spacy.load("de_core_news_sm")
# nlp_stanza_spacy = spacy_stanza.load_pipeline("de")
config = {
    'processors': 'tokenize,mwt,pos,lemma,depparse',  # Comma-separated list of processors to use
    'lang': 'de',  # Language code for the language to build the Pipeline in
    'tokenize_model_path': './de_gsd_models/de_gsd_tokenizer.pt',
    # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
    'mwt_model_path': './de_gsd_models/de_gsd_mwt_expander.pt',
    'pos_model_path': './de_gsd_models/de_gsd_tagger.pt',
    'pos_pretrain_path': './de_gsd_models/de_gsd.pretrain.pt',
    'lemma_model_path': './de_gsd_models/de_gsd_lemmatizer.pt',
    'depparse_model_path': './de_gsd_models/de_gsd_parser.pt',
    'depparse_pretrain_path': './de_gsd_models/de_gsd.pretrain.pt'
}
# nlp_stanford = stanfordnlp.Pipeline(**config)
# stanza.download('de')
nlp_stanza = stanza.Pipeline('de')
stopwords_list = get_stop_words('german') + get_stop_words('english') + stopwords.words(
    "english") + stopwords.words("german")


def part_of_speech():
    pos = []
    name_entity = []

    df = pd.DataFrame(list(messages_original.find())).head(2000)
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
    df.to_pickle('PoS.pkl')
    # print(df.head(20))
    # pprint.pprint(df['PoS'].values.tolist()[:200])

    return df


def aspect_based_opinion_mining():
    stop_words = set(stopwords.words('german'))
    # df = pd.read_pickle('PoS_1.pkl')
    df = pd.DataFrame(list(messages_original.find()))
    PATH_TO_JAR = 'C:/Users/budde/PycharmProjects/Masterthesis/Telegramgroups/stanford-ner-2020-11-17/stanford-ner-4.2.0.jar'
    PATH_TO_MODEL = 'C:/Users/budde/PycharmProjects/Masterthesis/Telegramgroups/stanford-ner-2020-11-17/classifiers/dewac_175m_600.crf.ser.gz'
    java_path = 'C:/Program Files/AdoptOpenJDK/jdk-11.0.11.9-hotspot'
    os.environ['JAVAHOME'] = java_path
    pos = part_of_speech()


    fcluster = []
    totalfeatureList = []
    finalcluster = []
    dic_cluster = {}
    dic = {}
    tagger = StanfordNERTagger(model_filename=PATH_TO_MODEL, path_to_jar=PATH_TO_JAR, encoding='utf-8')

    for text in df['Message']:
        nlp_message = nlp(text)
        taggedList = []
        for token in nlp_message:
            tagged_list = [token.text, token.tag_]
            taggedList.append(tagged_list)
        # print(taggedList)
        new_word_list = []
        flag = 0
        for i in range(0, len(taggedList) - 1):
            if taggedList[i][1] == "NN" and taggedList[i + 1][1] == "NN":
                new_word_list.append(taggedList[i][0] + taggedList[i + 1][0])
                flag = 1
            else:
                if flag == 1:
                    flag = 0
                    continue
                new_word_list.append(taggedList[i][0])
                if i == len(taggedList) - 2:
                    new_word_list.append(taggedList[i + 1][0])

        finaltxt = ' '.join(word for word in new_word_list)

        new_txt_list = nltk.word_tokenize(finaltxt)
        wordList = [w for w in new_txt_list if not w in stopwords_list]
        # print(wordList)
        pos_tag = []
        for i in wordList:
            word_list = nlp(i)
            for token in word_list:
                pos_list = [token.text, token.tag_]
                pos_tag.append(pos_list)

        doc = nlp_stanza(finaltxt)
        dep_node = []
        for dep_edge in doc.sentences[0].dependencies:
            dep_node.append([dep_edge[2].text, dep_edge[0].id, dep_edge[1]])

        for i in range(0, len(dep_node)):
            if int(dep_node[i][1]) != 0:
                dep_node[i][1] = new_word_list[(int(dep_node[i][1]) - 1)]
        print(dep_node)

        featureList = []
        categories = []
        for i in pos_tag:
            if i[1] == 'JJ' or i == "NN" or i[1] == "JJR" or i[1] == "NNS" or i[1] == "RB":
                featureList.append(list(i))
                totalfeatureList.append(list(i))
                categories.append(i[0])
        # print('featureList: ', featureList)

        for i in featureList:
            filist = []
            for j in dep_node:
                if ((j[0] == i[0] or j[1] == i[0]) and (
                        j[2] in ["nsubj", "acl:relcl", "obj", "dobj", "agent", "advmod", "amod", "neg", "prep_of",
                                 "acomp", "xcomp", "compound"])):
                    if j[0] == i[0]:
                        filist.append(j[1])
                    else:
                        filist.append(j[0])
                fcluster.append([i[0], filist])
            print("filist: ", filist)

    for i in totalfeatureList:
        dic[i[0]] = i[1]
    print("dic: ", dic)

    for i in fcluster:
        if dic[i[0]] == "NN":
            finalcluster.append(i)
            dic_cluster['NN'] = i

    finalcluster = [x for x in finalcluster if x[1]]

    print("total feature list:", totalfeatureList)
    print('final cluster', finalcluster)
    print("dic cluster", dic_cluster)

    # with open("finalcluster.pkl", "wb") as f:
    #    pickle.dump(finalcluster, f)

    # return finalcluster


def name_entity_rec():
    df = pd.read_pickle('PoS_1.pkl')
    PATH_TO_JAR = 'C:/Users/budde/PycharmProjects/Masterthesis/Telegramgroups/stanford-ner-2020-11-17/stanford-ner-4.2.0.jar'
    PATH_TO_MODEL = 'C:/Users/budde/PycharmProjects/Masterthesis/Telegramgroups/stanford-ner-2020-11-17/classifiers/dewac_175m_600.crf.ser.gz'
    java_path = 'C:/Program Files/AdoptOpenJDK/jdk-11.0.11.9-hotspot'
    os.environ['JAVAHOME'] = java_path
    # doc = nlp_stanza(df)

    tagger = StanfordNERTagger(model_filename=PATH_TO_MODEL, path_to_jar=PATH_TO_JAR, encoding='utf-8')
    sentiment = []
    for text in df['Message'].head(20):
        words = nltk.word_tokenize(text)
        pos_tag = nltk.pos_tag(words)
        tagged_words = tagger.tag(words)
        # print(tagged_words)
        blob = TextBlobDE(text)
        if tagged_words == "I-PER" or tagged_words == "B-PER" or tagged_words == "I-ORG" or tagged_words == "B-ORG":
            sentiment.append((tagged_words, ": ", blob.sentences, blob.noun_phrases, blob.sentiment))

    pprint.pprint(sentiment)


def nltk_sentiment():
    df = pd.read_pickle('PoS_1.pkl')

    for message in df['Message']:
        words = nltk.word_tokenize(message)
        all_words = nltk.FreqDist(words)

        word_features = list(all_words)[:2000]
        document = set(message)
        features = {}
        for word in word_features:
            features['contains({})'.format(word)] = (word in document)

        return features


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


if __name__ == '__main__':
    print(aspect_based_opinion_mining())
