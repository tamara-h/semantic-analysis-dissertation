import csv, time, random
import pandas
import numpy as np

from nltk.stem import WordNetLemmatizer

# Function to perform a binary search over the value rating dataset
def binary_search(array, target):
    lower = 0
    upper = len(array)
    while lower < upper:   # use < instead of <=
        x = lower + (upper - lower) // 2
        val = array[x][0]
        if target == val:
            return x
        elif target > val:
            if lower == x:
                break
            lower = x
        elif target < val:
            upper = x


def inputAnalysis(inputText):
    # Dataset which contains the VAD values for induvidual words
    lexicon = pandas.read_csv("../data/word_edited_raw.csv", sep=",")
    words_values = []
    words = []
    for index,row in lexicon.iterrows():
        words.append(row["sentence"])
        words_values.append(row)

    input_words = inputText.split(' ')
    v_sum = 0
    a_sum = 0
    d_sum = 0
    count = 0
    avgVAD = 0

    # for each word
    for i in input_words:
        #  go through each word value?

        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized_word = wordnet_lemmatizer.lemmatize(i)

        index = binary_search(words_values, lemmatized_word)
        if (index):
            # Calculate average VAD values for the sentence
            # Sum the values for each word
            v_sum += float(words_values[index][1])/2
            a_sum += float(words_values[index][2])/2
            d_sum += float(words_values[index][3])/2
            count += 1

        # If at least one of the words have been found
    if count > 0:
        # Avg VAD for the sentence
        v = round(v_sum / count)
        if v == 1:
            v = 2
        elif v == 5:
            v = 4
        a = round(a_sum / count)
        if a == 1:
            a = 2
        elif a == 5:
            a = 4

        d = round(d_sum / count)
        if d == 1:
            d = 2
        elif d == 5:
            d = 4
        avgVAD = [v, a, d]

        # print("avg 3d: " + str(avgVAD))
        return v, a, d
    else:
        return None,None,None


def getF1Score():
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score
    emoBank = pandas.read_csv("../data/emoBank2.csv", sep="\t")
    # emoBank = emoBank.head(500)

    v_true = []
    v_pred = []

    a_true = []
    a_pred = []

    d_true = []
    d_pred = []


    for index, row in emoBank.iterrows():
        # print((row))
        p_v, p_a, p_d = inputAnalysis(row['sentence'])

        if(p_v):
            v_true.append(round(row['Valence']))
            v_pred.append(p_v)

        if (p_a):
            a_true.append(round(row["Arousal"]))
            a_pred.append(p_a)

        if(p_d):
            d_true.append(round(row["Dominance"]))
            d_pred.append(p_d)



    print(confusion_matrix(v_true, v_pred))
    print(f1_score(v_true, v_pred, average="micro"))
    print(confusion_matrix(a_true, a_pred))
    print(f1_score(a_true, a_pred, average="micro"))
    print(confusion_matrix(d_true, d_pred))
    print(f1_score(d_true, d_pred, average="micro"))
    elapsed = time.time() - startTime

    print("Elapsed time: " + str(elapsed))


startTime = time.time()
getF1Score()