over = read.csv("~/Documents/fypFiles/tlh537/sentimentAnalysis/data/oversample.csv", header=TRUE, sep="\t");
under = read.csv("~/Documents/fypFiles/tlh537/sentimentAnalysis/data/undersample.csv", header=TRUE, sep="\t");

test1<-wilcox.test(over[,2],over[,5],alternative = "g",paired=FALSE, conf.level = 0.95)$p.value
