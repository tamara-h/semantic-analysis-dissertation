import pandas
import numpy as np
import time


# This file contains the processing data for ngram / n features tests and model selection


emoBank = pandas.read_csv("./data/emoBank2.csv", sep="\t")
# emoBankBinary = pandas.read_csv("./data/emoBank2.csv", sep="\t")

# print(emoBank.columns.values)
start_time = time.time()

x = emoBank.sentence
y = emoBank.Valence

def avgIgnore0(array):
    sum = 0
    length = 0
    for i in array:
        if i !=0:
            length += 1
            sum = sum + i

    return sum/length


def plotResults(unigram, bigram, trigram, fourgram):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(unigram.nfeatures, unigram.f1_score, label='unigram',color='gold')
    plt.plot(bigram.nfeatures, bigram.f1_score, label='bigram', color='orangered')
    plt.plot(trigram.nfeatures, trigram.f1_score, label='trigram', color='royalblue')
    plt.plot(fourgram.nfeatures, fourgram.f1_score, label='fourgram', color='brown')
    # plt.plot(fivegram.nfeatures, fivegram.f1_score, label='fivegram', color='violet')
    # plt.plot(sixgram.nfeatures, sixgram.f1_score, label='sixgram', color='lime')

    plt.title("N-gram(1~4) test result : f1_score with 4 classes")
    plt.xlabel("Number of features")
    plt.ylabel("F1_score")
    plt.legend()
    elapsed = time.time() - start_time
    print("overall time: " + str(elapsed))
    plt.show()



def stratifiedSplitFixed(ngram_range=(1,2), n_features=[85000]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import Pipeline

    ngramStartTime = time.time()
    result = []
    results_dataset = pandas.DataFrame()

    for n in n_features:
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=3000)

        f1_scores = []

        for train_index, test_index in sss.split(x, y):
            cvec = CountVectorizer()
            cvec.set_params(stop_words=None, max_features=n, ngram_range=ngram_range, encoding="utf-8")

            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = LogisticRegression(solver='liblinear', multi_class="auto")

            pipeline = Pipeline([
                ('vectorizer', cvec),
                ('classifier', clf)
            ])

            sentiment_fit = pipeline.fit(x_train, y_train)
            y_pred = sentiment_fit.predict(x_test)

            # conmat = np.array(confusion_matrix(y_test, y_pred, labels=[2.0, 3.0, 4.0]))

            calc_f1 = f1_score(y_test, y_pred, labels=[2.0, 3.0, 4.0], average="micro")
            # print(calc_f1)
            f1_scores.append(calc_f1)



        averageF1 = np.sum(f1_scores) / len(f1_scores)
        # print("Avg")
        # print(averageF1)

        result.append((n, averageF1))
        results_dataset[n] = pandas.Series(f1_scores)
    # print(result)
    pathString = "./data/nonBinary" + str(ngram_range[1]) + "Grams.csv"
    results_dataset.to_csv(pathString, sep="\t")
    elapsed = time.time() - ngramStartTime
    print("time for " + str(ngram_range[1]) + ": " + str(elapsed))
    return result



def stratifiedSplitVary():

    n_feature_range = np.arange(5000, 300001, 20000)

    feature_result_ug = stratifiedSplitFixed(ngram_range=(1, 1), n_features=n_feature_range)
    feature_result_bg = stratifiedSplitFixed(ngram_range=(1, 2), n_features=n_feature_range)
    feature_result_tg = stratifiedSplitFixed(ngram_range=(1, 3), n_features=n_feature_range)
    feature_result_fg = stratifiedSplitFixed(ngram_range=(1, 4), n_features=n_feature_range)
    # feature_result_fig = stratifiedSplitFixed(ngram_range=(1, 5), n_features=n_feature_range)
    # feature_result_sg = stratifiedSplitFixed(ngram_range=(1, 6), n_features=n_feature_range)

    nfeatures_plot_bg = pandas.DataFrame(feature_result_bg,
                                         columns=['nfeatures', 'f1_score'])
    nfeatures_plot_ug = pandas.DataFrame(feature_result_ug,
                                         columns=['nfeatures', 'f1_score'])
    nfeatures_plot_tg = pandas.DataFrame(feature_result_tg,
                                         columns=['nfeatures', 'f1_score'])
    nfeatures_plot_fg = pandas.DataFrame(feature_result_fg,
                                         columns=['nfeatures', 'f1_score'])

    plotResults(nfeatures_plot_ug, nfeatures_plot_bg, nfeatures_plot_tg, nfeatures_plot_fg)




def varyModel():
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB, BernoulliNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedShuffleSplit


    classifiers = [
        LogisticRegression(solver='liblinear', multi_class="auto"),
        LinearSVC(),
        MultinomialNB(),
        BernoulliNB(),
        KNeighborsClassifier(),
        RandomForestClassifier()
    ]

    cvec = CountVectorizer()

    def classifier_comparator(vectorizer=cvec, n_features=85000, ngram_range=(1, 2),
                              classifier=classifiers):

        classifiers = pandas.DataFrame()
        times = pandas.DataFrame()

        for c in classifier:

            result = []
            elapsedTime = []
            for train_index, test_index in sss.split(x, y):
                startModelTime = time.time()
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]


                vectorizer.set_params(max_features=n_features, ngram_range=ngram_range)
                pipeline = Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', c)
                ])

                sentiment_fit = pipeline.fit(x_train, y_train)
                y_pred = sentiment_fit.predict(x_test)

                f1 = f1_score(y_test, y_pred, labels=[2.0, 3.0, 4.0], average="micro")
                modelTime = time.time() - startModelTime
                elapsedTime.append(modelTime)
                result.append(f1)

            classifiers[type(c).__name__] = result
            times[type(c).__name__] = elapsedTime
            # print(classifiers.head())

        return classifiers, times

    def plotTrigramRes(classifiersDataframe):
        import matplotlib.pyplot as plt

        columnNames = []
        barValues = []

        for column in classifiersDataframe:

            columnNames.append(column)
            columnValue = 0
            count = 0
            for row in classifiersDataframe[column]:
                columnValue += row
                count += 1
            averageColScore = columnValue / count
            barValues.append(averageColScore)

        n_groups = len(columnNames)
        fig, ax = plt.subplots()

        index = np.arange(n_groups)
        opacity = 0.6

        ax.bar(index, barValues,
                        alpha=opacity, color='m',
                        label='n_features=85000, ngram_range=(1, 2)')

        ax.set_xlabel('Model')
        ax.set_ylabel('F1 Scores')
        ax.set_ylim([0.75, 0.79])
        ax.set_title('F1 Scores by model')
        ax.set_xticks(index)
        ax.set_xticklabels(columnNames)
        ax.legend()

        fig.tight_layout()
        plt.show()

    FOLDS = 10
    sss = StratifiedShuffleSplit(n_splits=FOLDS, test_size=0.2, random_state=3000)


    bigram_result, times = classifier_comparator(n_features=85000, ngram_range=(1, 2))

    results_dataset = pandas.DataFrame(data=bigram_result)
    pathString = "./data/nonBinaryModels.csv"
    results_dataset.to_csv(pathString, sep="\t")

    times_dataset = pandas.DataFrame(data=times)
    times_pathString = "./data/modelTimes.csv"
    times_dataset.to_csv(times_pathString, sep="\t")

    plotTrigramRes(bigram_result)




def main():
    # stratifiedSplitVary()
    varyModel()


main()