# https://medium.com/analytics-vidhya/aspect-based-sentiment-analysis-a-practical-approach-8f51029bbc4a
import pickle
from itertools import chain
import pprint
import pandas as pd
import stanfordnlp
import nltk
import os
from nltk.corpus import stopwords
import spacy
from spacy.tokens import Span
from Telegramgroups import NLP
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import numpy as np
from itertools import islice
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize.treebank import TreebankWordDetokenizer
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

    tagger = StanfordNERTagger(model_filename=PATH_TO_MODEL, path_to_jar=PATH_TO_JAR, encoding='utf-8')
    fcluster = []
    totalfeatureList = []
    final_cluster = []
    dic = {}
    parse_tree = []
    sentiment = []
    tagged_words_list = []
    entity_chunks = []
    feature_list = []
    for text in df['Message'].head(200):
        nlp_message = nlp(text)
        words = nltk.word_tokenize(text)
        tagged_words = tagger.tag(words)
        tagged_words_list.append(tagged_words)

        for token, tag in tagged_words:
            if tag != 'O':
                entity_chunks.append((token, tag))

        taggedList = []
        for token_nlp in nlp_message:
            tagged_list = [token_nlp.text, token_nlp.tag_]
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
                    new_word_list.append(taggedList[i][0])

        final_text = ' '.join(word for word in new_word_list)

        new_txt_list = word_tokenize(final_text)
        wordList = [w for w in new_txt_list if not w in stopwords_list]
        text_sentences = TreebankWordDetokenizer().detokenize(wordList)
        doc = nlp(text_sentences)
        for token in doc:
            head_child = [token.text, token.dep_, token.head.text, token.head.pos_]
            parse_tree.append(head_child)
            if token.head.pos_ == 'NOUN':
                feature_list.append([token.head.text, token.text])
            else:
                continue

    # print('1', feature_list)
    dic_cluster = pd.DataFrame(feature_list, columns=['Noun', 'features'])
    print(dic_cluster)
    # print("total feature list:", totalfeatureList)
    # print('final cluster', final_cluster)
    # print(dic_cluster)

def name_entity_sentiment():
    txt = aspect_based_opinion_mining()
    nlp.add_pipe("merge_entities")
    nlp.add_pipe("merge_noun_chunks")
    df = pd.DataFrame(txt)

    entity_list = []
    for text in txt['Noun']:
        doc = nlp(text)
        ents = [e.label_ for e in doc.ents]
        entity_list.append(ents)

    txt['Entity'] = entity_list

    # txt.to_csv('entity.csv')
    print(txt.iloc[10:60])


def entity():
    PATH_TO_JAR = 'C:/Users/budde/PycharmProjects/Masterthesis/Telegramgroups/stanford-ner-2020-11-17/stanford-ner-4.2.0.jar'
    PATH_TO_MODEL = 'C:/Users/budde/PycharmProjects/Masterthesis/Telegramgroups/stanford-ner-2020-11-17/classifiers/dewac_175m_600.crf.ser.gz'
    java_path = 'C:/Program Files/AdoptOpenJDK/jdk-11.0.11.9-hotspot'
    os.environ['JAVAHOME'] = java_path
    # doc = nlp_stanza(df)
    txt = pd.DataFrame(list(messages_original.find()))
    tagger = StanfordNERTagger(model_filename=PATH_TO_MODEL, path_to_jar=PATH_TO_JAR, encoding='utf-8')
    sentiment = []
    for text in txt['Message']:
        words = nltk.word_tokenize(text)
        tagged_words = tagger.tag(words)
        print(tagged_words)


if __name__ == '__main__':
    aspect_based_opinion_mining()
    # name_entity_sentiment()
