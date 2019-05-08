nFeaturesResults = read.csv("~/Documents/fypFiles/tlh537/sentimentAnalysis/data/nonBinary3Grams.csv", header=TRUE, sep="\t");

# try to find the value after which there is no significant increase in values

test1<-wilcox.test(nFeaturesResults[,2],nFeaturesResults[,11],alternative = "l",paired=FALSE, conf.level = 0.95)$p.value
# test1 = 8.880533e-05
# results show there is an increase, so we continue to refine
test2<-wilcox.test(nFeaturesResults[,3],nFeaturesResults[,11],alternative = "l",paired=FALSE, conf.level = 0.95)$p.value
# test2 = 0.0001184868
test3<-wilcox.test(nFeaturesResults[,4],nFeaturesResults[,11],alternative = "l",paired=FALSE, conf.level = 0.95)$p.value
# test3 = 0.003522
test4<-wilcox.test(nFeaturesResults[,5],nFeaturesResults[,11],alternative = "l",paired=FALSE, conf.level = 0.95)$p.value
# test4 = 0.04759
test5<-wilcox.test(nFeaturesResults[,6],nFeaturesResults[,11],alternative = "l",paired=FALSE, conf.level = 0.95)$p.value
# test5 =  0.1713
# This p value is greater than 5% so we can say there is no significant increase in F1 value after this point