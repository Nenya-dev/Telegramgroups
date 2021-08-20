def name_entity_rec():
    for text in txt['Noun']:
        doc = nlp(text)
        if doc[0].ent_type_:
            txt['Entity'] = doc[0].ent_type_
        else:
            txt['Entity'] = None
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
            if tagged_words == "I-PER" or tagged_words == "B-PER" or tagged_words == "I-ORG" or tagged_words == "B-ORG":
                for elem in txt['features']:
                    sentiment.append((tagged_words, ": ", elem, ', ', TextBlobDE(elem).sentiment))

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

def abom():
    for token_nlp in nlp_message:
        tagged_list = [token_nlp.text, token_nlp.tag_]
        taggedList.append(tagged_list)

    new_word_list = []
    flag = 0
    for i in range(0, len(taggedList) - 1):
        if taggedList[i][1] == "NN" and taggedList[i + 1][1] == "NN":
            new_word_list.append(taggedList[i][0] + ' ' + taggedList[i + 1][0])
            flag = 1
        else:
            if flag == 1:
                flag = 0
                continue
            new_word_list.append(taggedList[i][0])
            if i == len(taggedList) - 2:
                new_word_list.append(taggedList[i][0])

                new_txt_list = word_tokenize(finaltxt)
                wordList = [w for w in new_txt_list if not w in stopwords_list]
                pos_tag = []
                for i in wordList:
                    word_list = nlp(i)
                    for token_nlp in word_list:
                        pos_list = [token_nlp.text, token_nlp.tag_]
                        pos_tag.append(pos_list)

                doc = nlp(finaltxt)

    finaltxt = ' '.join(word for word in new_word_list)
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

    final_cluster = [x for x in final_cluster if x[1]]