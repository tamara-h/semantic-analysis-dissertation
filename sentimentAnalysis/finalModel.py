import pandas
import numpy as np

emoBank = pandas.read_csv("./data/emoBank2.csv", sep="\t")
emoBankBinary = pandas.read_csv("./data/emoBank2.csv", sep="\t")

x = emoBank.sentence
v = emoBank.Valence
a = emoBank.Arousal
d = emoBank.Dominance

nullacc = 0

def modelF1(ngram_range=(1,2), n_features=85000, dimension="v"):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import f1_score
    from imblearn.pipeline import make_pipeline

    FOLDS = 10

    sss = StratifiedShuffleSplit(n_splits=FOLDS, test_size=0.2, random_state=3000)
    result = []

    if (dimension == "v"):
        y = v
    elif dimension == "a":
        y = a
    else:
        y = d

    for train_index, test_index in sss.split(x, y):
        cvec = CountVectorizer()
        cvec.set_params(max_features=n_features, ngram_range=ngram_range)
        clf = MultinomialNB()

        pipeline = make_pipeline(cvec, clf)

        # X = cvec.fit_transform(x)
        x_train, x_test = x[train_index], x[test_index]
        # X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)

        f1 = f1_score(y_test, y_pred, labels=[2.0, 3.0, 4.0], average="micro")
        result.append(f1)

    avgScore = 0
    for score in result:
        avgScore += score

    elapsedTime = time.time() - start_time
    print("elapsed time: " + str(elapsedTime))

    return avgScore/FOLDS


import time
start_time = time.time()


print(modelF1(dimension="v"))
print(modelF1(dimension="a"))
print(modelF1(dimension="d"))