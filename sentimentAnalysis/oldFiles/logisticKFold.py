import pandas
import numpy as np

emoBank = pandas.read_csv("./data/emoBank.csv", sep="\t")

# x = emoBank.drop(columns=["id","sentence","Valence"], inplace=False).values
# x = emoBank.drop(columns=["id","sentence"], inplace=False).values
x = emoBank.sentence
y = emoBank.Valence


from sklearn.model_selection import train_test_split
# SEED = 3000

nullacc = 0

# Test out stratified split with a fixed bigram and number of features
def stratifiedSplitFixed(ngram_range=(1,3), n_features=[200000]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import Pipeline

    result = []
    for n in n_features:
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=4000)
        confusion_sum = [[0,0],[0,0]]

        for train_index, test_index in sss.split(x, y):
            cvec = CountVectorizer()
            cvec.set_params(stop_words=None, max_features=n, ngram_range=ngram_range)
            # X = cvec.fit_transform(x)
            # print(train_index)
            y_train, y_test = y[train_index], y[test_index]
            x_train, x_test = x[train_index], x[test_index]
            # X_train, X_test = X[train_index], X[test_index]

            clf = LogisticRegression(solver='liblinear')

            pipeline = Pipeline([
                ('vectorizer', cvec),
                ('classifier', clf)
            ])

            sentiment_fit = pipeline.fit(x_train, y_train)
            y_pred = sentiment_fit.predict(x_test)

            # print("Im so sad this is actually very upsetting I hate my life")
            # print(x_test.shape)

            acc = accuracy_score(y_test, y_pred)

            conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0, 1]))
            confusion_sum = confusion_sum + conmat

            # print("Classification Report\n")
            # print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

            # print(confusion)

        confusion = pandas.DataFrame(confusion_sum, index=['negative', 'positive'],
                                        columns=['predicted_negative', 'predicted_positive'])
        true_neg = confusion_sum[0][0]
        false_pos = confusion_sum[0][1]
        false_neg = confusion_sum[1][0]
        true_pos = confusion_sum[1][1]
        # print(confusion)

        precision = (true_pos)/(true_pos+false_pos)
        recall = (true_pos) / (true_pos + false_neg)

        f1 = 2 * (precision*recall) / (precision + recall)
        print(f1)
        result.append((n,f1))

    # print(result)
    return result



def plotResults(unigram, bigram, trigram, fourgram):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(unigram.nfeatures, unigram.f1_score, label='unigram',color='gold')
    plt.plot(bigram.nfeatures, bigram.f1_score, label='bigram', color='orangered')
    # plt.plot(bigramTF.nfeatures, bigramTF.f1_score, label='bigram',linestyle=':', color='orangered')
    plt.plot(trigram.nfeatures, trigram.f1_score, label='trigram', color='royalblue')
    # plt.plot(trigramTF.nfeatures, trigramTF.f1_score, label='trigramTF',linestyle=':', color='royalblue')
    plt.plot(fourgram.nfeatures, fourgram.f1_score, label='fourgram', color='brown')
    # plt.plot(fivegram.nfeatures, fivegram.f1_score, label='fivegram', color='violet')
    # plt.plot(sixgram.nfeatures, sixgram.f1_score, label='sixgram', color='lime')
    # plt.plot(sevengram.nfeatures, sevengram.f1_score, label='sevengram', color='turquoise')

    # plt.plot(unigramTF.nfeatures, unigramTF.f1_score, label='unigram', linestyle=':',color='gold')
    # plt.plot([10000, 100000], [nullacc, nullacc], label='zero accuracy')
    plt.title("N-gram(1~3) test result : f1_score")
    plt.xlabel("Number of features")
    plt.ylabel("f1_score")
    plt.legend()
    elapsed = time.time() - start_time
    print("time: " + str(elapsed))
    plt.show()



def stratifiedSplitVary():
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    tvec = CountVectorizer()

    n_feature_range = np.arange(10000, 300001, 20000)

    feature_result_ug = stratifiedSplitFixed(ngram_range=(1,1), n_features=n_feature_range)
    feature_result_bg = stratifiedSplitFixed(ngram_range=(1,2), n_features=n_feature_range)
    feature_result_tg = stratifiedSplitFixed(ngram_range=(1,3), n_features=n_feature_range)
    feature_result_fg = stratifiedSplitFixed(ngram_range=(1,4), n_features=n_feature_range)
    # feature_result_fig = stratifiedSplitFixed(ngram_range=(1,5), n_features=n_feature_range)
    # feature_result_sg = stratifiedSplitFixed(ngram_range=(1,6), n_features=n_feature_range)
    # feature_result_seg = stratifiedSplitFixed(ngram_range=(1,7), n_features=n_feature_range)

    nfeatures_plot_bg = pandas.DataFrame(feature_result_bg,
                                         columns=['nfeatures', 'f1_score'])
    nfeatures_plot_ug = pandas.DataFrame(feature_result_ug,
                                         columns=['nfeatures', 'f1_score'])
    nfeatures_plot_tg = pandas.DataFrame(feature_result_tg,
                                         columns=['nfeatures', 'f1_score'])
    nfeatures_plot_fg = pandas.DataFrame(feature_result_fg,
                                         columns=['nfeatures', 'f1_score'])
    # nfeatures_plot_fig = pandas.DataFrame(feature_result_fig,
    #                                      columns=['nfeatures', 'f1_score'])
    #
    # nfeatures_plot_sg = pandas.DataFrame(feature_result_sg,
    #                                      columns=['nfeatures', 'f1_score'])
    # nfeatures_plot_seg = pandas.DataFrame(feature_result_seg,
    #                                      columns=['nfeatures', 'f1_score'])



    feature_result_ugTF = []
    feature_result_bgTF = []
    feature_result_tgTF = []

    # The below is for testing with the other vectoriser

    # feature_result_ugTF = stratifiedSplitFixed(ngram_range=(1, 1), n_features=n_feature_range)
    # feature_result_bgTF = stratifiedSplitFixed(ngram_range=(1, 2), n_features=n_feature_range)
    # feature_result_tgTF = stratifiedSplitFixed(ngram_range=(1, 3), n_features=n_feature_range)

    # nfeatures_plot_tgTF = pandas.DataFrame(feature_result_tgTF,
    #                                      columns=['nfeatures', 'f1_score'])
    # nfeatures_plot_bgTF = pandas.DataFrame(feature_result_bgTF,
    #                                      columns=['nfeatures', 'f1_score'])
    # nfeatures_plot_ugTF = pandas.DataFrame(feature_result_ugTF,
    #                                      columns=['nfeatures', 'f1_score'])

    # plotResults(nfeatures_plot_ug,nfeatures_plot_bg,nfeatures_plot_tg, nfeatures_plot_ugTF, nfeatures_plot_bgTF, nfeatures_plot_tgTF);
    plotResults(nfeatures_plot_ug, nfeatures_plot_bg, nfeatures_plot_tg, nfeatures_plot_fg)


#
def varyModel():
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB, BernoulliNB
    from sklearn.linear_model import RidgeClassifier
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors import NearestCentroid
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedShuffleSplit

    # names = ["Logistic Regression", "Linear SVC", "LinearSVC with L1-based feature selection", "Multinomial NB",
    #          "Bernoulli NB", "Ridge Classifier", "AdaBoost", "Perceptron", "Passive-Aggresive", "Nearest Centroid", "K Nearest Neighbours", "Random Forest"]
    # Set tol to 1e-3 because documentation says that this is what it defaults to, and specifying it silences warnings
    classifiers = [
        LogisticRegression(solver='liblinear'),
        LinearSVC(),
        Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
            ('classification', LinearSVC(penalty="l2"))]),
        MultinomialNB(),
        BernoulliNB(),
        RidgeClassifier(),
        AdaBoostClassifier(),
        Perceptron(tol=1e-3, max_iter=5),
        PassiveAggressiveClassifier(tol=1e-3, max_iter=5),
        NearestCentroid(),
        KNeighborsClassifier(),
        RandomForestClassifier()
    ]
    # zipped_clf = zip(names, classifiers)

    cvec = CountVectorizer()

    def classifier_comparator(vectorizer=cvec, n_features=0, stop_words=None, ngram_range=(1, 1),
                              classifier=classifiers, x_train=[], x_test=[], y_train=[], y_test=[]):
        result = []
        vectorizer.set_params(stop_words=stop_words, max_features=n_features, ngram_range=ngram_range)
        for c in classifier:
            pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', c)
            ])

            sentiment_fit = pipeline.fit(x_train, y_train)
            y_pred = sentiment_fit.predict(x_test)

            conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0, 1]))

            confusion = pandas.DataFrame(conmat, index=['negative', 'positive'],
                                         columns=['predicted_negative', 'predicted_positive'])
            true_neg = conmat[0][0]
            false_pos = conmat[0][1]
            false_neg = conmat[1][0]
            true_pos = conmat[1][1]
            # print(confusion)

            precision = (true_pos) / (true_pos + false_pos)
            recall = (true_pos) / (true_pos + false_neg)

            f1 = 2 * (precision * recall) / (precision + recall)
            result.append((type(c).__name__, precision, recall, f1))
        return result

    def plotTrigramRes(result1, result2):
        import matplotlib.pyplot as plt

        n_groups = len(result1)
        labels = []
        # print(n_groups)

        f1_1 = []
        f1_2 = []
        for i in result1:
            labels.append(i[0])
            f1_1.append(i[3])

        for i in result2:
            f1_2.append(i[3])

        fig, ax = plt.subplots()

        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.4

        rects1 = ax.bar(index, f1_1, bar_width,
                        alpha=opacity, color='b',
                        label='100000')

        rects2 = ax.bar(index + bar_width, f1_2, bar_width,
                        alpha=opacity, color='r',
                        label='200000')

        ax.set_xlabel('Model')
        ax.set_ylabel('F1 Scores')
        ax.set_ylim([0.84, 0.94])
        ax.set_title('F1 Scores by model and n-gram values')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(labels)
        ax.legend()

        fig.tight_layout()
        plt.show()

    FOLDS = 1
    sss = StratifiedShuffleSplit(n_splits=FOLDS, test_size=0.2, random_state=3000)
    resSum1 = []
    resSum2 = []
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        trigram_result = classifier_comparator(n_features=100000, ngram_range=(1, 3), x_train=x_train,
                                               x_test=x_test, y_train=y_train, y_test=y_test)
        trigram_result2 = classifier_comparator(n_features=10, ngram_range=(1, 3), x_train=x_train,
                                                x_test=x_test, y_train=y_train, y_test=y_test)

        for i in trigram_result:
            found = False
            for j in resSum1:
                if i[0] == j[0]:
                    found = True
                    j[1] += i[1]
                    j[2] += i[2]
                    j[3] += i[3]
            if not found:
                resSum1.append(list(i))

        for i in trigram_result2:
            found = False
            for j in resSum2:
                if i[0] == j[0]:
                    found = True
                    j[1] += i[1]
                    j[2] += i[2]
                    j[3] += i[3]
            if not found:
                resSum2.append(list(i))

    # print("before divide")
    # print(resSum1)
    for i in resSum1:
        i[1] = i[1]/FOLDS
        i[2] = i[2]/FOLDS
        i[3] = i[3]/FOLDS
    for i in resSum2:
        i[1] = i[1]/FOLDS
        i[2] = i[2]/FOLDS
        i[3] = i[3]/FOLDS
    # print("******************")
    # print("after divide")
    # print(resSum1)

    plotTrigramRes(resSum1, resSum2)


import time
start_time = time.time()
stratifiedSplitVary()
# stratifiedSplitFixed()
# varyModel()
# countVectorize()