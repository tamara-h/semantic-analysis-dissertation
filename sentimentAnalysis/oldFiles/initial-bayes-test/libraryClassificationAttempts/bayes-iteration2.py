#           ITERATION 2
# This iteration just uses valance values and classifies them in buckets

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import csv, random


with open("../data/raw.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    letter_text=[]
    for row in tsvreader:
        letter_text.append(row)
with open("../data/reader.tsv") as tsvfile:
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

sentences = []
def word_feats(words):
    return dict([(word, True) for word in words])

sentencefeats =[(word_feats(f[0].split()), int(round(f[1]))) for f in sentence_valence]
random.shuffle(sentencefeats)
print(sentencefeats[0])



random.shuffle(sentencefeats)

cutoff = len(sentencefeats) * 3 / 4

trainfeats = sentencefeats[:int(cutoff)]
testfeats = sentencefeats[int(cutoff):]


print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()