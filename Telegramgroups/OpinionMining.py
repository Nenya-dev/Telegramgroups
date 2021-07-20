import pandas as pd
from textblob_de import TextBlobDE as tb
import nltk
from nltk.corpus import stopwords
import spacy
from Telegramgroups import NLP
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import numpy as np
from itertools import islice
posts = NLP.posts_prep_two


def part_of_speech():
    pos = []
    name_entity = []

    nlp = spacy.load("de_core_news_sm")
    df = pd.DataFrame(list(posts.find()))
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

    return df


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


def aspect_sentiment_analysis():
    stopwords_list = stopwords.words("german")
    fcluster = []
    totalfeatureList = []
    finalcluster = []
    dic = []

    sentList = nltk.sent_tokenize(txt)

    for line in sentList:
        newtaggedList = []
        txt_list = nltk.word_tokenize(line)
        taggedList = nltk.pos_tag(txt_list)

        newwordList = []
        flag = 0
        for i in range(0, len(taggedList) - 1):
            if taggedList[i][1] == 'NN' and taggedList[i + 1][1] == 'NN':
                newwordList.append(taggedList[i][0] + taggedList[i + 1][0])
                flag = 1
            else:
                if flag == 1:
                    flag = 0
                    continue
                newwordList.append(taggedList[i][0])
                if i == len(taggedList) - 2:
                    newwordList.append(taggedList[i + 1][0])

        finaltxt = ' '.join(word for word in newwordList)
        new_txt_list = nltk.word_tokenize(finaltxt)
        wordsList = [w for w in new_txt_list if not w in stopwords_list]
        taggedList_word = nltk.pos_tag(wordsList)

        doc = nlp(finaltxt)  # Object of Stanford NLP Pipeleine

        # Getting the dependency relations betwwen the words
        dep_node = []
        for dep_edge in doc.sentences[0].dependencies:
            dep_node.append([dep_edge[2].text, dep_edge[0].index, dep_edge[1]])

        # Coverting it into appropriate format
        for i in range(0, len(dep_node)):
            if int(dep_node[i][1]) != 0:
                dep_node[i][1] = newwordList[(int(dep_node[i][1]) - 1)]

        featureList = []
        categories = []
        for i in taggedList:
            if i[1] == 'JJ' or i[1] == 'NN' or i[1] == 'JJR' or i[1] == 'NNS' or i[1] == 'RB':
                featureList.append(list(i))  # For features for each sentence
                totalfeatureList.append(list(i))  # Stores the features of all the sentences in the text
                categories.append(i[0])

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

    for i in totalfeatureList:
        dic[i[0]] = i[1]

    for i in fcluster:
        if dic[i[0]] == "NN":
            finalcluster.append(i)

    return finalcluster


if __name__ == '__main__':
   # part_of_speech()
#   feature_extraction()
    TFidf()
