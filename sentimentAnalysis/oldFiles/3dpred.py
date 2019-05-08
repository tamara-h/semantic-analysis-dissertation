import pandas
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

emoBank = pandas.read_csv("./data/emoBank2.csv", sep="\t")

x = emoBank.sentence
y = emoBank.Valence
xAdjust = emoBank.drop(columns=["id", "sentence", "Valence"], inplace=False).values


sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=4000)

for train_index, test_index in sss.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    xA_train, xA_test = xAdjust[train_index], xAdjust[test_index]


def avgIgnore0(array):
    sum = 0
    length = 0
    for i in array:
        if i !=0:
            length += 1
            sum = sum + i

    return sum/length


def f1FromConMat(confusion_sum):
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
    print("Precision and recall")
    print(precision_recall)
    print("F1 scores")
    print(f1_scores)
    print("Average F1")
    print(avgIgnore0(f1_scores))

from sklearn.model_selection import train_test_split
SEED = 3000

def trainSentenceModel():
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import CountVectorizer
    from imblearn.pipeline import make_pipeline
    from imblearn.over_sampling import SMOTE


    cvec = CountVectorizer()
    cvec.set_params(stop_words=None, max_features=200000, ngram_range=(1,3))
    clf = LogisticRegression(solver='liblinear', multi_class="auto")
    SMOTE_pipeline = make_pipeline(cvec, SMOTE(random_state=SEED), clf)

    sentiment_fit = SMOTE_pipeline.fit(x_train, y_train)
    return sentiment_fit


def scoreAdjust(dimension="Valence"):
    from sklearn.linear_model import LogisticRegression
    from imblearn.pipeline import make_pipeline
    from imblearn.over_sampling import SMOTE

    clf = LogisticRegression(solver='liblinear', multi_class="auto")
    SMOTE_pipeline = make_pipeline(SMOTE(random_state=SEED), clf)

    sentiment_fit = SMOTE_pipeline.fit(xA_train, y_train)
    return sentiment_fit


def ensembleClassification():
    from sklearn.metrics import confusion_matrix

    valenceSentenceModel = trainSentenceModel()
    valenceADModel = scoreAdjust("Valence")

    # classifier = VotingClassifier(estimators=[("string", valenceSentenceModel),("string1", valenceADModel)])

    y_pred_sentence = valenceSentenceModel.predict(x_test)
    # print(xA_test)
    # print(xA_test[0])
    y_pred_adjust = valenceADModel.predict(xA_test)

    
    combined_pred = []
    
    for i in range(len(y_pred_sentence)):
        if y_pred_sentence[i] == y_pred_adjust[i]:
            combined_pred.append(y_pred_sentence[i])
        else:
            difference = y_pred_sentence[i] - y_pred_adjust[i]
            if(difference > 1):
                newPred = y_pred_sentence[i] - (difference/2)
                combined_pred.append(newPred)
            else:
                combined_pred.append(y_pred_sentence[i])

    # print(len(y_pred_sentence))
    # print(len(y_pred_adjust))
    # print(combined_pred)


    # conmat = np.array(confusion_matrix(y_test, y_pred_sentence, labels=[2.0, 3.0, 4.0]))
    # print(conmat)
    # f1FromConMat(conmat)
    #
    # conmat1 = np.array(confusion_matrix(y_test, y_pred_adjust, labels=[2.0, 3.0, 4.0]))
    # print(conmat1)
    # f1FromConMat(conmat1)

    conmat_combined = np.array(confusion_matrix(y_test, combined_pred, labels=[2.0, 3.0, 4.0]))
    print(conmat_combined)
    f1FromConMat(conmat_combined)



import time
start_time = time.time()

# trainSentenceModel()
# scoreAdjust()

ensembleClassification()