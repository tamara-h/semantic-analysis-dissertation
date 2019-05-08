nFeaturesResults = read.csv("~/Documents/fypFiles/tlh537/sentimentAnalysis/data/nonBinary3Grams.csv", header=TRUE, sep="\t");

# try to find the value after which there is no significant increase in values
# comparing to 165,000 features since it is the highest on the graph
test1<-wilcox.test(nFeaturesResults[,2],nFeaturesResults[,10],alternative = "l",paired=FALSE, conf.level = 0.95)$p.value
# results show there is an increase, so we continue to refine
test2<-wilcox.test(nFeaturesResults[,3],nFeaturesResults[,10],alternative = "l",paired=FALSE, conf.level = 0.95)$p.value

test3<-wilcox.test(nFeaturesResults[,4],nFeaturesResults[,10],alternative = "l",paired=FALSE, conf.level = 0.95)$p.value

test4<-wilcox.test(nFeaturesResults[,5],nFeaturesResults[,10],alternative = "l",paired=FALSE, conf.level = 0.95)$p.value

test5<-wilcox.test(nFeaturesResults[,6],nFeaturesResults[,10],alternative = "l",paired=FALSE, conf.level = 0.95)$p.value
# This p value is greater than 5% so we can say there is no significant increase in F1 value after this point