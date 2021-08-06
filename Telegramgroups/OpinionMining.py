# https://medium.com/analytics-vidhya/aspect-based-sentiment-analysis-a-practical-approach-8f51029bbc4a
import pickle

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

MODELS_DIR = '.'
# stanfordnlp.download('de', MODELS_DIR)
posts = NLP.posts_prep_two
messages_original = NLP.posts_prep_one
nlp = spacy.load("de_core_news_sm")
nlp_stanza_spacy = spacy_stanza.load_pipeline("de")
config = {
    'processors': 'tokenize,mwt,pos,lemma,depparse',  # Comma-separated list of processors to use
    'lang': 'fr',  # Language code for the language to build the Pipeline in
    'tokenize_model_path': './de_gsd_models/de_gsd_tokenizer.pt',
    # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
    'mwt_model_path': './de_gsd_models/de_gsd_mwt_expander.pt',
    'pos_model_path': './de_gsd_models/de_gsd_tagger.pt',
    'pos_pretrain_path': './de_gsd_models/de_gsd.pretrain.pt',
    'lemma_model_path': './de_gsd_models/de_gsd_lemmatizer.pt',
    'depparse_model_path': './de_gsd_models/de_gsd_parser.pt',
    'depparse_pretrain_path': './de_gsd_models/de_gsd.pretrain.pt'
}
nlp_stanford = stanfordnlp.Pipeline(**config)
# stanza.download('de')
nlp_stanza = stanza.Pipeline('de')


def part_of_speech():
    pos = []
    name_entity = []

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
    df.to_pickle('PoS_2.pkl')

    return df


def aspect_based_opinion_mining():
    stop_words = set(stopwords.words('german'))
    df = pd.read_pickle('PoS_1.pkl')
    PATH_TO_JAR = 'C:/Users/budde/PycharmProjects/Masterthesis/Telegramgroups/stanford-ner-2020-11-17/stanford-ner-4.2.0.jar'
    PATH_TO_MODEL = 'C:/Users/budde/PycharmProjects/Masterthesis/Telegramgroups/stanford-ner-2020-11-17/classifiers/dewac_175m_600.crf.ser.gz'
    java_path = 'C:/Program Files/AdoptOpenJDK/jdk-11.0.11.9-hotspot'
    os.environ['JAVAHOME'] = java_path


    fcluster = []
    totalfeatureList = []
    finalcluster = []
    dic_cluster = {}
    dic = {}
    tagger = StanfordNERTagger(model_filename=PATH_TO_MODEL, path_to_jar=PATH_TO_JAR, encoding='utf-8')

    try:
        for message in df['Message'].head(200):
            sentList = nltk.sent_tokenize(message)
            for text in sentList:
                words = nltk.word_tokenize(text)
                pos_tag = nltk.pos_tag(words)
                tagged = tagger.tag(words)

                newwordList = []
                flag = 0
                for i in range(0, len(pos_tag) - 1):
                    if pos_tag[i][1] == "NN" and pos_tag[i + 1][1] == "NN":
                        newwordList.append(pos_tag[i][0] + pos_tag[i + 1][0])
                        flag = 1
                    else:
                        if flag == 1:
                            flag = 0
                            continue
                        newwordList.append(pos_tag[i][0])
                        if i == len(pos_tag) - 2:
                            newwordList.append(pos_tag[i + 1][0])

                finaltxt = ' '.join(word for word in newwordList)
                # print(finaltxt)

                new_txt_list = nltk.word_tokenize(finaltxt)
                wordList = [w for w in new_txt_list if not w in stop_words]
                pos_tag = nltk.pos_tag(wordList)

                doc = nlp_stanza(finaltxt)
                # doc = nlp_stanza_spacy(finaltxt)
                dep_node = []
                for dep_edge in doc.sentences[0].dependencies:
                    dep_node.append([dep_edge[2].text, dep_edge[0].id, dep_edge[1]])
                    # print(dep_node)

                for i in range(0, len(dep_node)):
                    # print(len(dep_node))
                    if int(dep_node[i][1]) != 0:
                        # print(dep_node[i][1], ":", newwordList[(int(dep_node[i][1]) - 1)])
                        dep_node[i][1] = newwordList[(int(dep_node[i][1]) - 1)]


                featureList = []
                categories = []
                for i in pos_tag:
                    if i[1] == 'JJ' or i == "NN" or i[1] == "JJR" or i[1] == "NNS" or i[1] == "RB":
                        featureList.append(list(i))
                        totalfeatureList.append(list(i))
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
                    dic_cluster['NN'] = i

            finalcluster = [x for x in finalcluster if x[1]]

            print("total feature list:", totalfeatureList)
            print('final cluster', finalcluster)
            print("dic cluster", dic_cluster)

       # with open("finalcluster.pkl", "wb") as f:
        #    pickle.dump(finalcluster, f)

        return finalcluster
    except IndexError as error:
        return print(error)


def sentiment():
    finalcluster = aspect_based_opinion_mining()


def name_entity_rec():
    df = pd.read_pickle('PoS_1.pkl')
    PATH_TO_JAR = 'C:/Users/budde/PycharmProjects/Masterthesis/Telegramgroups/stanford-ner-2020-11-17/stanford-ner-4.2.0.jar'
    PATH_TO_MODEL = 'C:/Users/budde/PycharmProjects/Masterthesis/Telegramgroups/stanford-ner-2020-11-17/classifiers/dewac_175m_600.crf.ser.gz'
    java_path = 'C:/Program Files/AdoptOpenJDK/jdk-11.0.11.9-hotspot'
    os.environ['JAVAHOME'] = java_path

    tagger = StanfordNERTagger(model_filename=PATH_TO_MODEL, path_to_jar=PATH_TO_JAR, encoding='utf-8')
    for text in df['Message'].head(20):
        words = nltk.word_tokenize(text)
        pos_tag = nltk.pos_tag(text)
        tagged = tagger.tag(words)
        print(tagged)
        print(pos_tag)


def word_dependency():
    # df = pd.DataFrame(list(messages_original.find()))
    df = pd.read_pickle('PoS_1.pkl')
    # print(df['PoS'].head(50))
    df1 = df.head(100)
    nlp.add_pipe("merge_entities")
    nlp.add_pipe("merge_noun_chunks")

    noun_chunks = {}
    for text in df['Message']:
        doc = nlp(text)
        if doc.ents:
            for ent in doc.ents:
                noun_chunks['Entity_Label'] = ent.label_
                noun_chunks['Entity_Text'] = ent.text
                for chunk in doc.noun_chunks:
                    noun_chunks['Text'] = chunk.text
                    noun_chunks['Root_Text'] = chunk.root.text
                    noun_chunks['Dep'] = chunk.root.dep_
                    noun_chunks['Head_Text'] = chunk.root.head.text
            print(noun_chunks)

    return noun_chunks


def sentiment():
    noun_chunks = word_dependency()


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
    print(aspect_based_opinion_mining())
