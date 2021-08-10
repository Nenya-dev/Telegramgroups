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

config = {
    'processors': 'tokenize,mwt,pos,lemma,depparse',  # Comma-separated list of processors to use
    'lang': 'de',  # Language code for the language to build the Pipeline in
    'tokenize_model_path': 'Telegramgroups/de_gsd_models/de_gsd_tokenizer.pt',
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
nlp_stanza_spacy = spacy_stanza.load_pipeline("de")
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

    fcluster = []
    totalfeatureList = []
    final_cluster = []
    dic_cluster = {}
    dic = {}
    for text in df['Message'].head(200):
        nlp_message = nlp(text)
        taggedList = []
        for token in nlp_message:
            tagged_list = [token.text, token.tag_]
            taggedList.append(tagged_list)

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

        new_txt_list = word_tokenize(finaltxt)
        wordList = [w for w in new_txt_list if not w in stopwords_list]
        pos_tag = []
        for i in wordList:
            word_list = nlp(i)
            for token in word_list:
                pos_list = [token.text, token.tag_]
                pos_tag.append(pos_list)
        # print(pos_tag)
        dep_node = []

        doc = nlp(finaltxt)
        for dep_edge in doc:
            dep_node.append([dep_edge.text, dep_edge.head.text, dep_edge.dep_])

        featureList = []
        categories = []
        for i in pos_tag:
            if i[1] == 'ADJA' or i[1] == "NN" or i[1] == "ADJD" or i[1] == "ADV":
                featureList.append(list(i))
                totalfeatureList.append(list(i))
                categories.append(i[0])

        for i in featureList:
            filist = []
            for j in dep_node:
                if ((j[0] == i[0] or j[1] == i[0]) and (
                        j[2] in ["sb", "sbp", "app", "oc", "og", "op", "oa", "avc", "adc",
                                 "ng", "mo", "pnc", "nk"])):
                    if j[0] == i[0]:
                        filist.append(j[1])
                    else:
                        filist.append(j[0])
            fcluster.append([i[0], filist])

    for i in totalfeatureList:
        dic[i[0]] = i[1]

    for i in fcluster:
        if dic[i[0]] == "NN":
            final_cluster.append(i)

    # final_cluster = [x for x in final_cluster if x[1]]
    dic_cluster = pd.DataFrame(final_cluster)
    # print("total feature list:", totalfeatureList)
    print('final cluster', final_cluster)
    print(('dic_cluster', dic_cluster))
    # print("dic cluster", dic_cluster)

    # with open("finalcluster.pkl", "wb") as f:
    #    pickle.dump(finalcluster, f)

    # return finalcluster


if __name__ == '__main__':
    print(aspect_based_opinion_mining())
