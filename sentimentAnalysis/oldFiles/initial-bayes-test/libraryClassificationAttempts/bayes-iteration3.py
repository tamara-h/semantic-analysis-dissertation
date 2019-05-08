#           ITERATION 3
# Incorporating other dataset

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import csv, random



with open("./data/raw.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    letter_text=[]
    for row in tsvreader:
        letter_text.append(row)
with open("./data/reader.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    letter_values=[]
    for row in tsvreader:
        letter_values.append(row)

sentence_valence = []

for i in range(len(letter_text)):
    for j in range(len(letter_values)):
        if letter_text[i][0] == letter_values[j][0]:
            sentence_valence.append([letter_text[i][1], float(letter_values[j][3])])

negSentences = []
posSentences = []

random.shuffle(sentence_valence)

for i in range(len(sentence_valence)):
    if sentence_valence[i][1] < 2.5:
        negSentences.append(sentence_valence[i][0])
    else:
        posSentences.append(sentence_valence[i][0])




# ADD OTHER DATASET

with open("./data/word_edited_raw.csv") as csvfile:
    csvreader = csv.reader(csvfile, delimiter=",")
    word_text=[]
    for row in csvreader:
        word_text.append(row)

word_valence = []

for i in range(len(word_text)):
    word_valence.append([word_text[i][0], float(word_text[i][1])])

for i in range(len(word_valence)):
    if word_valence[i][1] <= 5:
        negSentences.append(word_valence[i][0])
    else:
        posSentences.append(word_valence[i][0])


def word_feats(words):
    return dict([(word, True) for word in words])

negfeats = [(word_feats(f.split()), 'neg') for f in negSentences]
posfeats = [(word_feats(f.split()), 'pos') for f in posSentences]


random.shuffle(negfeats)
random.shuffle(posfeats)

negcutoff = len(negfeats) * 3 / 4
poscutoff = len(posfeats) * 3 / 4

trainfeats = negfeats[:int(negcutoff)] + posfeats[:int(poscutoff)]
testfeats = negfeats[int(negcutoff):] + posfeats[int(poscutoff):]


print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()