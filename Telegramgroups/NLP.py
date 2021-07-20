# https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/
import numpy as np
from nltk import word_tokenize
from pymongo import MongoClient
import pandas as pd
from collections import Counter
import pymongo
import configparser
import nltk
from HanTa import HanoverTagger as ht
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import AllTelegramGroups
from nltk.corpus import stopwords
import re
import string
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from stop_words import get_stop_words

# nltk.download()
parser = configparser.ConfigParser()
config_path = r'config.ini'
parser.read(config_path)
client_db = pymongo.MongoClient(parser['mongoDB']["client"])
db = client_db['Telegram_Test']
collection = db["Querdenker_Test"]
posts_prep_one = db["Querdenker_Test_Prep_1"]
posts_prep_two = db["Querdenker_Test_Prep_2"]
query = {'_id': 0, 'ID': 0, 'Group': 0, 'Fwd_group': 0, 'Channel': 0, 'Forwarded': 0, 'Date': 0}


def mongoDB_to_dataframe():
    text_data = pd.DataFrame(list(collection.find()))
    # print(text_data['Message'])
    df = text_data['Message'].convert_dtypes(convert_string=True)
    # print(df.head(10))
    return df


tagger = ht.HanoverTagger('morphmodel_ger.pgz')


def data_prep():
    df = mongoDB_to_dataframe()
    df.dropna(inplace=True)
    for message in df:
        message = str(message).lower()  # Lowercase words
        message = re.sub(r"\[(.*?)\]", "", message)  # Remove [+XYZ chars] in content
        # message = re.sub(r"[^a-zA-Z0-9üäöÜÄÖß]", "", message)  # Remove [+XYZ chars] in content
        message = re.sub(r"\s+", " ", message)  # Remove multiple spaces in content
        message = re.sub(r"\w+…|…", "", message)  # Remove ellipsis (and last word)
        message = re.sub(r"(?<=\w)-(?=\w)", " ", message)  # Replace dash between words
        message = re.sub(r'http\S+|www.\S+', "", message)  # Remove urls
        message = re.sub(r"(?<=\w)-(?=\w)", " ", message)  # Replace dash between words
        # message = re.sub(r"[^\w\s]", "", message)  # Remove punctuation
        message = re.sub(r"[!@#$%^&*()[]{};:,./<>?\|`~-=_+__]", "", message)
        message = re.sub(r"\W+", " ", message)
        message = re.sub("["
                         u"\U0001F600-\U0001F64F"  # emoticons
                         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                         u"\U0001F680-\U0001F6FF"  # transport & map symbols
                         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                         u"\U00002702-\U000027B0"
                         u"\U000024C2-\U0001F251"
                         u"\U0001f926-\U0001f937"
                         u"\u200d"
                         u"\u2640-\u2642"
                         "]+", '', message)

        messages_dict = {'Message': message}
        posts_prep_one.insert_one(messages_dict)


def tokenize():
    text_data = pd.DataFrame(list(posts_prep_one.find()))
    df = text_data['Message'].convert_dtypes(convert_string=True)
    lemmatizer = WordNetLemmatizer()
    stopwords_list = get_stop_words('german') + get_stop_words('english') + stopwords.words(
        "english") + stopwords.words("german")

    for message in df:
        word_tokens = word_tokenize(message)
        # print(word_tokens)
        remove_stopwords = [w for w in word_tokens if not w in stopwords_list]

        clean_messages = []
        for word in remove_stopwords:
            word = lemmatizer.lemmatize(word)
            clean_messages.append(word)
        # print(clean_messages)

        message_prep = TreebankWordDetokenizer().detokenize(clean_messages)
        message_dict = {"Message": message_prep}
        posts_prep_two.insert_one(message_dict)

        # print(message_prep.head(10))


def frequency_distribution():
    text_data = pd.DataFrame(list(posts_prep_two.find()))
    df = text_data['Message']
    word_count = Counter(" ".join(df).split()).most_common(20)
    word_frequency = pd.DataFrame(word_count, columns=['Word', 'Frequency'])
    print(word_frequency)
    # plt.show()


if __name__ == '__main__':
    mongoDB_to_dataframe()
    data_prep()
    tokenize()
    frequency_distribution()
