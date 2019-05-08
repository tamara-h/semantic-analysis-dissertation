modelResults = read.csv("~/Documents/fypFiles/tlh537/sentimentAnalysis/data/nonBinaryModels.csv", header=TRUE, sep="\t");


Logistic_Multinomial<-wilcox.test(modelResults[,2],modelResults[,7],alternative = "two.sided",paired=FALSE, conf.level = 0.95)$p.value
Logisitc_RandomForest<-wilcox.test(modelResults[,2],modelResults[,4],alternative = "two.sided",paired=FALSE, conf.level = 0.95)$p.value
Multinomial_RandomForest<-wilcox.test(modelResults[,4],modelResults[,7],alternative = "two.sided",paired=FALSE, conf.level = 0.95)$p.value
meanLog <- avg(modelResults[,2])