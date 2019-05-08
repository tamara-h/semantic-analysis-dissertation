uniGram = read.csv("~/Documents/fypFiles/tlh537/sentimentAnalysis/data/nonBinary1Grams.csv", header=TRUE, sep="\t");
biGram = read.csv("~/Documents/fypFiles/tlh537/sentimentAnalysis/data/nonBinary2Grams.csv", header=TRUE, sep="\t");
triGram = read.csv("~/Documents/fypFiles/tlh537/sentimentAnalysis/data/nonBinary3Grams.csv", header=TRUE, sep="\t");
# try to find the value after which there is no significant increase in values

uniBi<-wilcox.test(uniGram[,6],biGram[,6],alternative = "l",paired=TRUE, conf.level = 0.95)$p.value

biTri<-wilcox.test(biGram[,6],triGram[,6],alternative = "l",paired=TRUE, conf.level = 0.95)$p.value
