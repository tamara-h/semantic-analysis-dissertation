import csv

with open("./data/word_edited_raw.csv") as csvfile:
    csvreader = csv.reader(csvfile, delimiter=",")
    word_text=[]
    for row in csvreader:
        word_text.append(row)

sentence_valence = []

for i in range(len(word_text)):
    sentence_valence.append([word_text[i][0], float(word_text[i][1])])

sentences = []
# block01 = []
# block12 = []
# block23 = []
# block34 = []
# block45 = []
#
# for i in sentence_valence:
#     if i[1] <=2:
#         block01.append(i)
#     elif 2<i[1]<=4:
#         block12.append(i)
#     elif 4<i[1]<=6:
#         block23.append(i)
#     elif 6<i[1]<=8:
#         block34.append(i)
#     elif 8<i[1]<=10:
#         block45.append(i)
#
# print(len(block01))
# print(len(block12))
# print(len(block23))
# print(len(block34))
# print(len(block45))



block0 = []
block1 = []
block2 = []
block3 = []
block4 = []
block5 = []
block6 = []
block7 = []
block8 = []
block9 = []

for i in sentence_valence:
    if i[1] <1:
        block0.append(i)
    elif 1<=i[1]<2:
        block1.append(i)
    elif 2<=i[1]<3:
        block2.append(i)
    elif 3<i[1]<=4:
        block3.append(i)
    elif 4<i[1]<=5:
        block4.append(i)
    elif 5<i[1]<=6:
        block5.append(i)
    elif 6<i[1]<=7:
        block6.append(i)
    elif 7<i[1]<=8:
        block7.append(i)
    elif 8<i[1]<=9:
        block8.append(i)
    elif 9<i[1]<=10:
        block9.append(i)

print(len(block0))
print(len(block1))
print(len(block2))
print(len(block3))
print(len(block4))
print(len(block5))
print(len(block6))
print(len(block7))
print(len(block8))
print(len(block9))

# print(len(block0)+len(block1)+len(block2)+len(block3)+len(block4))

#
# sentences = []
# block05 = []
# block510 = []
#
# for i in sentence_valence:
#     if i[1] <=5:
#         block05.append(i)
#     else:
#         block510.append(i)
#
# print(len(block05))
# print(len(block510))