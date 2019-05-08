import pandas
import numpy as np

emoBank = pandas.read_csv("./data/emoBank2.csv", sep="\t")
emoBankBinary = pandas.read_csv("./data/emoBank2.csv", sep="\t")

# print(emoBank.columns.values)

x = emoBank.sentence
y = emoBank.Valence

# print(x.head())

def avgIgnore0(array):
    sum = 0
    length = 0
    for i in array:
        if i !=0:
            length += 1
            sum = sum + i

    return sum/length



from sklearn.model_selection import train_test_split
# SEED = 3000

nullacc = 0

def stratifiedSplitFixed3Class(ngram_range=(1,3), n_features=[200000]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import Pipeline

    result = []
    for n in n_features:
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=3000)
        confusion_sum = [[0,0,0],[0,0,0],[0,0,0]]

        for train_index, test_index in sss.split(x, y):
            cvec = CountVectorizer()
            cvec.set_params(stop_words=None, max_features=n, ngram_range=ngram_range, encoding="utf-8")
            # X = cvec.fit_transform(x)
            x_train, x_test = x[train_index], x[test_index]
            # X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = LogisticRegression(solver='liblinear')

            pipeline = Pipeline([
                ('vectorizer', cvec),
                ('classifier', clf)
            ])

            sentiment_fit = pipeline.fit(x_train, y_train)
            y_pred = sentiment_fit.predict(x_test)


            # print(x_test.shape)

            acc = accuracy_score(y_test, y_pred)
            # print(acc)
            # print(y_test)

            conmat = np.array(confusion_matrix(y_test, y_pred, labels=[2.0, 3.0, 4.0]))
            # print(conmat)
            confusion_sum = confusion_sum + conmat


            # print("Classification Report\n")
            # print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

            # print(confusion)

        print(confusion_sum)


        def calculatePrecision(i):
            true_pos = confusion_sum[i][i]
            if (true_pos == 0):
                return 0
            else:
                sumRow = 0
                for j in confusion_sum[i]:
                    sumRow = sumRow + j
                precision = true_pos/sumRow
                return precision
        def calculateRecall(i):
            true_pos = confusion_sum[i][i]
            if(true_pos == 0):
                return 0
            else:
                sumCol = 0
                for j in confusion_sum:
                    sumCol = sumCol + j[i]
                recall = true_pos/sumCol
                return recall

        precision_recall = []
        f1_scores = []
        for i in range(len(confusion_sum)):
            precison = calculatePrecision(i)
            recall = calculateRecall(i)
            precision_recall.append([precison, recall])
            if (precison+recall) == 0:
                f1_scores.append(0)
            else:
                f1_scores.append(2 * (precison * recall) / (precison + recall))
        print(precision_recall)
        print(f1_scores)
        print("Average F1")
        print(avgIgnore0(f1_scores))


def overSample(ngram_range=(1,3), n_features=[200000]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import confusion_matrix
    from imblearn.pipeline import make_pipeline
    from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler

    result = []
    for n in n_features:
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=3000)
        confusion_sum = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        for train_index, test_index in sss.split(x, y):
            cvec = CountVectorizer()
            SEED = 777
            cvec.set_params(stop_words=None, max_features=n, ngram_range=ngram_range)
            clf = LogisticRegression(solver='liblinear')

            ROS_pipeline = make_pipeline(cvec, RandomOverSampler(random_state=SEED), clf)
            SMOTE_pipeline = make_pipeline(cvec, SMOTE(random_state=SEED), clf)
            ADASYN_pipeline = make_pipeline(cvec, ADASYN(ratio='minority', random_state=SEED), clf)


            # X = cvec.fit_transform(x)
            x_train, x_test = x[train_index], x[test_index]
            # X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            sentiment_fit = SMOTE_pipeline.fit(x_train, y_train)
            y_pred = sentiment_fit.predict(x_test)

            conmat = np.array(confusion_matrix(y_test, y_pred, labels=[2.0, 3.0, 4.0]))

            confusion_sum = confusion_sum + conmat

        print(confusion_sum)

        def calculatePrecision(i):
            true_pos = confusion_sum[i][i]
            if (true_pos == 0):
                return 0
            else:
                sumRow = 0
                for j in confusion_sum[i]:
                    sumRow = sumRow + j
                precision = true_pos / sumRow
                return precision

        def calculateRecall(i):
            true_pos = confusion_sum[i][i]
            if (true_pos == 0):
                return 0
            else:
                sumCol = 0
                for j in confusion_sum:
                    sumCol = sumCol + j[i]
                recall = true_pos / sumCol
                return recall

        precision_recall = []
        f1_scores = []
        for i in range(len(confusion_sum)):
            precison = calculatePrecision(i)
            recall = calculateRecall(i)
            precision_recall.append([precison, recall])
            if (precison + recall) == 0:
                f1_scores.append(0)
            else:
                f1_scores.append(2 * (precison * recall) / (precison + recall))
        print(precision_recall)
        print(f1_scores)
        print("Average F1")
        print(avgIgnore0(f1_scores))



def underSample(ngram_range=(1,3), n_features=[200000]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import confusion_matrix
    from imblearn.under_sampling import NearMiss, RandomUnderSampler
    from imblearn.pipeline import make_pipeline


    result = []
    for n in n_features:
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=3000)
        confusion_sum = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

        for train_index, test_index in sss.split(x, y):
            cvec = CountVectorizer()
            SEED = 777
            cvec.set_params(stop_words=None, max_features=n, ngram_range=ngram_range)
            clf = LogisticRegression(solver='liblinear')

            # RUS_pipeline = make_pipeline(cvec, RandomUnderSampler(random_state=777), clf)
            NM1_pipeline = make_pipeline(cvec, NearMiss(ratio='not minority', random_state=777, version=1), clf)
            NM2_pipeline = make_pipeline(cvec, NearMiss(ratio='not minority', random_state=777, version=2), clf)
            NM3_pipeline = make_pipeline(cvec, NearMiss(ratio='not minority', random_state=777, version=3, n_neighbors_ver3=4),clf)
            # nm3 = NearMiss(ratio='not minority', random_state=777, version=3, n_neighbors_ver3=4)


            # X = cvec.fit_transform(x)
            x_train, x_test = x[train_index], x[test_index]
            # X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            sentiment_fit = NM3_pipeline.fit(x_train, y_train)
            y_pred = sentiment_fit.predict(x_test)

            conmat = np.array(confusion_matrix(y_test, y_pred, labels=[1.0, 2.0, 3.0, 4.0, 5.0]))

            confusion_sum = confusion_sum + conmat

        print(confusion_sum)

        def calculatePrecision(i):
            true_pos = confusion_sum[i][i]
            if (true_pos == 0):
                return 0
            else:
                sumRow = 0
                for j in confusion_sum[i]:
                    sumRow = sumRow + j
                precision = true_pos / sumRow
                return precision

        def calculateRecall(i):
            true_pos = confusion_sum[i][i]
            if (true_pos == 0):
                return 0
            else:
                sumCol = 0
                for j in confusion_sum:
                    sumCol = sumCol + j[i]
                recall = true_pos / sumCol
                return recall

        precision_recall = []
        f1_scores = []
        for i in range(len(confusion_sum)):
            precison = calculatePrecision(i)
            recall = calculateRecall(i)
            precision_recall.append([precison, recall])
            if (precison + recall) == 0:
                f1_scores.append(0)
            else:
                f1_scores.append(2 * (precison * recall) / (precison + recall))
        print(precision_recall)
        print(f1_scores)

        print("Average F1")
        print(avgIgnore0(f1_scores))
        # f1 = 2 * (precision*recall) / (precision + recall)
        # print(f1)
        # result.append((n,f1))



def overSampleBinary(ngram_range=(1,3), n_features=[200000]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import confusion_matrix
    from imblearn.pipeline import make_pipeline
    from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler

    result = []
    for n in n_features:
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=3000)
        confusion_sum = [[0, 0], [0, 0]]

        for train_index, test_index in sss.split(x, y):
            cvec = CountVectorizer()
            SEED = 777
            cvec.set_params(stop_words=None, max_features=n, ngram_range=ngram_range)
            clf = LogisticRegression(solver='liblinear')

            SMOTE_pipeline = make_pipeline(cvec, SMOTE(random_state=SEED), clf)

            # X = cvec.fit_transform(x)
            x_train, x_test = x[train_index], x[test_index]
            # X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            sentiment_fit = SMOTE_pipeline.fit(x_train, y_train)
            y_pred = sentiment_fit.predict(x_test)

            conmat = np.array(confusion_matrix(y_test, y_pred, labels=[2.0, 3.0, 4.0]))

            confusion_sum = confusion_sum + conmat

        print(confusion_sum)

        def calculatePrecision(i):
            true_pos = confusion_sum[i][i]
            if (true_pos == 0):
                return 0
            else:
                sumRow = 0
                for j in confusion_sum[i]:
                    sumRow = sumRow + j
                precision = true_pos / sumRow
                return precision

        def calculateRecall(i):
            true_pos = confusion_sum[i][i]
            if (true_pos == 0):
                return 0
            else:
                sumCol = 0
                for j in confusion_sum:
                    sumCol = sumCol + j[i]
                recall = true_pos / sumCol
                return recall

        precision_recall = []
        f1_scores = []
        for i in range(len(confusion_sum)):
            precison = calculatePrecision(i)
            recall = calculateRecall(i)
            precision_recall.append([precison, recall])
            if (precison + recall) == 0:
                f1_scores.append(0)
            else:
                f1_scores.append(2 * (precison * recall) / (precison + recall))
        print(precision_recall)
        print(f1_scores)
        print("Average F1")
        print(avgIgnore0(f1_scores))



import time
start_time = time.time()

stratifiedSplitFixed3Class()
# overSample()
# countVectorize()
# underSample()