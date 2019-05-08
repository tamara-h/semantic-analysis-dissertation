import pandas
import numpy as np
import time
from sklearn.naive_bayes import MultinomialNB
from imblearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

emoBank = pandas.read_csv("./data/emoBank2.csv", sep="\t")

# x = emoBank.sentence
yV = emoBank.Valence
yA = emoBank.Arousal
yD = emoBank.Dominance
SEED = 777


def adjust(dimension=yV, dimName="Valence"):
    FOLDS = 10
    sss = StratifiedShuffleSplit(n_splits=FOLDS, test_size=0.2, random_state=3000)
    result = []
    x = emoBank.drop(columns=["id", "sentence", dimName], inplace=False).values
    for train_index, test_index in sss.split(x, dimension):

        clf = MultinomialNB()
        pipeline = make_pipeline(clf)

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = dimension[train_index], dimension[test_index]

        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        f1 = f1_score(y_test, y_pred, labels=[2.0, 3.0, 4.0], average="micro")
        result.append(f1)

    avgScore = 0
    for score in result:
        avgScore += score

    elapsedTime = time.time() - start_time
    print("elapsed time: " + str(elapsedTime))

    print("F1 score for " + str(dimName) + ": " + str(avgScore / FOLDS))

    return sentiment_fit

start_time = time.time()
def main():
    start_time = time.time()

    v_adj_array = np.empty((2065,))
    a_adj_array = np.empty((2065,))
    d_adj_array = np.empty((2065,))
    v_adj_array = [[-1, -1] for x in range(v_adj_array.size)]
    a_adj_array = [[-1, -1] for x in range(a_adj_array.size)]
    d_adj_array = [[-1, -1] for x in range(d_adj_array.size)]

    vAdjModel = adjust(dimension=yV, dimName="Valence")
    aAdjModel = adjust(dimension=yA, dimName="Arousal")
    dAdjModel = adjust(dimension=yD, dimName="Dominance")

main();