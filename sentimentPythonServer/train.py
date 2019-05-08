import numpy as np
import pandas
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

emoBank = pandas.read_csv("./data/emoBank.csv", sep="\t")

x = emoBank.sentence
yV = emoBank.Valence
yA = emoBank.Arousal
yD = emoBank.Dominance
SEED = 777

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=3000)
trainInd = None
testInd = None
for train_index, test_index in sss.split(x, yV):
   trainInd = train_index
   testInd = test_index

def train(dimension=yV):
    # Valence Train
    # Get rid of for loop
    cvec = CountVectorizer()
    cvec.set_params(stop_words=None, max_features=85000, ngram_range=(1,2))
    clf = MultinomialNB()

    pipeline = make_pipeline(cvec, clf)

    x_train, x_test = x[trainInd], x[testInd]
    y_train, y_test = dimension[trainInd], dimension[testInd]

    sentiment_fit = pipeline.fit(x_train, y_train)
    return sentiment_fit


def adjust(dimension=yV, dimName="Valence"):

    clf = MultinomialNB()
    pipeline = make_pipeline(clf)

    x = emoBank.drop(columns=["id", "sentence", dimName], inplace=False).values

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = dimension[train_index], dimension[test_index]

    sentiment_fit = pipeline.fit(x_train, y_train)
    return sentiment_fit


valenceModel = train(dimension=yV)
arousalModel = train(dimension=yA)
dominanceModel = train(dimension=yD)
vAdjModel = adjust(dimension=yV, dimName="Valence")
aAdjModel = adjust(dimension=yA, dimName="Arousal")
dAdjModel = adjust(dimension=yD, dimName="Dominance")

def adjustedPrediction(sentencePred, adjPred):
    print(sentencePred)
    print(adjPred)
    if sentencePred == adjPred:
        return sentencePred
    else:
        difference = sentencePred - adjPred
        newPred = sentencePred - (difference / 2)
        return newPred


def predict(words):
    test_array = np.empty((2065,))
    v_adj_array = np.empty((2065,))
    a_adj_array = np.empty((2065,))
    d_adj_array = np.empty((2065,))
    test_array = ["" for x in range(test_array.size)]
    v_adj_array = [[-1, -1] for x in range(v_adj_array.size)]
    a_adj_array = [[-1, -1] for x in range(a_adj_array.size)]
    d_adj_array = [[-1, -1] for x in range(d_adj_array.size)]
    test_array[0] = words


    vprediction = valenceModel.predict(test_array)
    vPred = vprediction[0].item() / 5

    aprediction = arousalModel.predict(test_array)
    aPred = aprediction[0].item() / 5
    # print(aPred)
    #
    dprediction = dominanceModel.predict(test_array)
    dPred = dprediction[0].item() / 5
    # print(dPred)

    v_adj_array[0] = [aPred,dPred]
    a_adj_array[0] = [vPred,dPred]
    d_adj_array[0] = [aPred,vPred]

    vAdjust = vAdjModel.predict(v_adj_array)
    aAdjust = aAdjModel.predict(a_adj_array)
    dAdjust = dAdjModel.predict(d_adj_array)

    vPred = adjustedPrediction(vprediction[0].item(), vAdjust[0].item())/5
    aPred = adjustedPrediction(aprediction[0].item(), aAdjust[0].item())/5
    dPred = adjustedPrediction(dprediction[0].item(), dAdjust[0].item())/5

    return vPred, aPred, dPred
