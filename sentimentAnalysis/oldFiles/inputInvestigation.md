

*/ USING PRECISION AND RECALL VALUES IN A TABLE /*

##N-Gram Variation

* Hypothesis: an increasing value of n will increase the f1 score, up to a point.

Instead of taking each word individually, by using an n-gram method to evaluate the dataset, we can take into account the surrounding words within the sentence.
By varying the n value here, we will investigate how a varying value for n affects the f1 score for the model.

The base to work off is to use a logistic regression model, (why this one) using unigrams to represent the collection of words in the dataset, 
where each word is not at all affected by its neighbours within a sentence. 

As we can see from the findings here that this gives an initial maximum f1 score reached of 0.9303547963206307.

As the value for n increases, we can see that f1 score does also increase as the number of features also increases. 
The difference between the results becomes very narrow after n is increased to 3, and since the computation time is notably increased (back this up?) between trigram and fourgram, we can use the trigram for the basis of other analysis over the model.
For the final chosen model we can choose to use the fourgram model, as the increase in score by using the six or seven value for n is negligible.

Using the Wilcoxon signed-rank test we can show that after x gram the data is the same when rounded to 3 decimal points. We have rounded to 3 decimal points because 



****** lets put the hypothesis test in here ********

## Number of features variation 
 
 The number of features in this case is the top words ordered by term frequency across the corpus.
 */INCLUDE THE ANALYSIS OF THE DATASET / WORD FREQUENCIES HERE/* 