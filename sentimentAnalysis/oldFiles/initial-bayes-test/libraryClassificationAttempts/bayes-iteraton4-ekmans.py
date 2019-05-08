#           ITERATION 4
# Redo initial classification by finding shortest distance and classifying as one of ekmans 6 basic emotions.

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import csv, random
import math


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

ek_anger = [-0.51, 0.59, 0.25]
ek_disgust = [-0.6, 0.35, 0.11]
ek_fear = [-0.64, 0.6, -0.43]
ek_happy = [0.81, 0.51, 0.46]
ek_sad = [-0.63, -0.27, 0.33]
ek_surprise = [0.4, 0.67, -0.13]


def ekman_adjust(emotion):
    for i in range(len(emotion)):
        emotion[i] = 2.5 * (1+emotion[i])
    return emotion


ek_emotions = [["anger", ekman_adjust(ek_anger)], ["disgust", ekman_adjust(ek_disgust)],
               ["fear", ekman_adjust(ek_fear)], ["happy", ekman_adjust(ek_happy)],
               ["sad", ekman_adjust(ek_sad)], ["surprise", ekman_adjust(ek_surprise)]]


sentence_vad = []

def shortest_dist(point):
    closest_emotion = -1
    shortest_dist = 5
    for i in ek_emotions:
            distance = math.sqrt((point[0] - i[1][0])**2 + (point[1] - i[1][1])**2 + (point[2] - i[1][2])**2)
            if distance < shortest_dist:
                shortest_dist = distance
                closest_emotion = i[0];

    return closest_emotion


sentence_emotion = []


for i in range(len(letter_text)):
    for j in range(len(letter_values)):
        if letter_text[i][0] == letter_values[j][0]:
            # Get VAD values for the sentence
            # sentence_vad.append([letter_text[i][1], float(letter_values[j][3]), float(letter_values[j][1]), float(letter_values[j][2])])
            v = float(letter_values[j][3])
            a = float(letter_values[j][1])
            d = float(letter_values[j][2])

            sentence_emotion.append([letter_text[i][1], shortest_dist([v, a, d])])
            # Find shortest distance between the point and the emotions

anger_count = []
dis_count = []
fear_count = []
happy_count = []
sad_count = []
surprise_count = []

def word_feats(words):
    return dict([(word, True) for word in words])

for i in sentence_emotion:
    i[0] = word_feats(i[0].split())
    if i[1] == "anger":
        anger_count.append(i)
    elif i[1] == "disgust":
        dis_count.append(i)
    elif i[1] == "fear":
        fear_count.append(i)
    elif i[1] == "happy":
        happy_count.append(i)
    elif i[1] == "sad":
        sad_count.append(i)
    elif i[1] == "surprise":
        surprise_count.append(i)
    else:
        print("BIG ERROR")

random.shuffle(anger_count)
random.shuffle(dis_count)
random.shuffle(fear_count)
random.shuffle(happy_count)
random.shuffle(sad_count)
random.shuffle(surprise_count)

anger_cutoff = len(anger_count) * 3 / 4
dis_cutoff = len(dis_count) * 3 / 4
fear_cutoff = len(fear_count) * 3 / 4
happy_cutoff = len(happy_count) * 3 / 4
sad_cutoff = len(sad_count) * 3 / 4
surprise_cutoff = len(surprise_count) * 3 / 4


trainfeats = anger_count[:int(anger_cutoff)] + dis_count[:int(dis_cutoff)] \
             + fear_count[:int(fear_cutoff)] + happy_count[:int(happy_cutoff)]\
             + sad_count[:int(sad_cutoff)] + surprise_count[:int(surprise_cutoff)]

testfeats = anger_count[int(anger_cutoff):] + dis_count[int(dis_cutoff):] \
             + fear_count[int(fear_cutoff):] + happy_count[int(happy_cutoff):]\
             + sad_count[int(sad_cutoff):] + surprise_count[int(surprise_cutoff):]

random.shuffle(trainfeats)
print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()