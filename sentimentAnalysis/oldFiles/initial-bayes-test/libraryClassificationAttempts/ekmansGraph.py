import csv

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

sentences = []

block01 = []
block12 = []
block23 = []
block34 = []
block45 = []
block56 = []
block67 = []
block78 = []
block89 = []
block910 = []

for i in sentence_valence:
    if i[1] <=0.5:
        block01.append(i)
    elif 0.5<i[1]<=1:
        block12.append(i)
    elif 1<i[1]<=1.5:
        block23.append(i)
    elif 1.5<i[1]<=2:
        block34.append(i)
    elif 2<i[1]<=2.5:
        block45.append(i)
    elif 2.5<i[1]<=3:
        block56.append(i)
    elif 3<i[1]<=3.5:
        block67.append(i)
    elif 3.5<i[1]<=4:
        block78.append(i)
    elif 4<i[1]<=4.5:
        block89.append(i)
    elif 4.5<i[1]<=5:
        block910.append(i)

print(len(block01))
print(len(block12))
print(len(block23))
print(len(block34))
print(len(block45))
print(len(block56))
print(len(block67))
print(len(block78))
print(len(block89))
print(len(block910))

