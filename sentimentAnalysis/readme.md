# Sentiment Analysis

This part of the project will use data from the EmoBank dataset and try to predict the VAD values for input text.


## Development Log
### Nov 5th
Push already completed code and complete a Bayes classifier for the corpus.
This is training on 6990 sentences and testing on 2330, returning an accuracy of 59.6%, which is far from ideal. I'm not shuffling the data yet which is a next step to see if it generally improves. Looking forward I think I will try other classifiying algorithms, this needs more research. I will take a step back from development for the time being and do more research.

* Added update to shuffle data, and lead to dramatic increase in accuracy, about 73-75%.
* At the moment its just classifying based on if the valance is <2.5, then its a negative statement, and if its >2.5 then its positive. I need to next incorporate the actual valance values into the training.
* Bayes (and most classifyers it looks like) dont like assigning numerical values which are pretty subjective. Where I could move forward from here is class them into the Ekmans 6 basic emotions with the dominance and arousal values


### Nov 6th
Since I'm working with continuous independent variables, I'll have to look at the algorithms more. Today is spent in research.
* Discretising for Bayes? into 1/2/3/4/5 - but none of my data is really classified into 1 or 5
* Do I need more data?

### Nov 12th
Looking into SMOTE and Support Vector Machines (SVM)

### Nov 13th
* Start a log of every time the program runs. 
* Test against 100% negative and 100% postive sentences.
* My worst enemy tried to kill me and they ruined my life.
* I fell in love, got married and have five beautiful children who all love me.
* Bag of words

### Nov 16th
* Add new dataset
* SMOTE, SVM
* balance out data, do testing with new stuff
* Starting iteration 3, going to put new dataset in and only rate on word valuation. 

### Nov 18th
* Tried analysing with new dataset, got terrible results, but thats to be expected because theyre just words, it wont really work.
* These need to be put into the training set for the sentences. 
* Restructuring data

### Nov 19th
###Things to Do this week
* Decide whether to tokenise each word
* Different classification
    * Ekmans
    * How many buckets
    * Pos neg
    
### December 3rd
* word tokenisation
* hey its classification that I actually recognise
* doing it by VAD

### December 4th
* Takes a LONG time to run, 15-20 seconds for 1000 records
* trying to optimise today
* creating new csv with the formatted EmoBank data so dont have to do it every time
* ran with linear search 20x, averaging 15 second running time
* implementing binary search
* binary search now gets average running time down to 0.1s, much better! (for 1000 records)
* 2.5s for the whole dataset instead of 2+ mins


* now want to look more into predicition accuracies, see how I can properly compare my methods
* tokenising words down to their fundamentals:
    * So words like "working" might not be in the dataset, but "work" is
    * im fairly sure theres a library for this
* using the Mean Absolute Percentage Error for the results
* over the dataset, the average MAPE values for VAD are (0.132, 0.254, 0.115)
* use the wordnet lemmatization to get roots of the words
* need to put words if theyre verbs n stuff into the lemma thing


### December 5th
* comparing ML methods
* keep track of random seeds for comparison and evaluation - prime numbers

### January 20th
* Going back to Bag of Words, put in negation?

### January 21st
* Doing some data visualisation
* text visualisation 
* found a large part of the text which has been misentered, sorting this out
* due to punctuation, quotation messes stuff up.

### January 22nd
* Word Frequencies 
* Zipfs law
* Instead of removing stop words, im going to keep them for now, since they may have an effect. 
* Can do a comparison before and after 

### January 23rd 
* created bokeh plot of valuating harmonic mean of pos/ neg 
* doesnt look right, and theres lots of stop words all over the place
* they gotta go
* comparisons of bokeh plot graphs

### January 24th
* Need to write up what exactly the bokeh plots are showing
* Darker the point is, the more negative it is
* Lets get onto some splitting
* Will just split arbitrarily for train/test
* Will perform k fold validation on in future because the dataset is so small
* Remember to shuffle!!

### January 28th
* Moving forward from last supervison
* Need to define the problem more
* Harmonic means.
* For simplicities sake, I will only consider quadrants in 3D
* using positive and negatives

### January 29th
* approx. 87% pos valence
* 96% pos arousal and dominance
* will use bayes to try and balance
* will focus on valence
* will use k fold with forced balances for now

### January 31st 
* RESEARCH QUESTIONS
    * Using Machine learning techniques over the Emobank dataset to predict a sentiment
    * Using oversampling techniques to deal with large class imbalance
    * How to properly predict data when there is a large class imbalance
    * Does using 2 more dimensions provide valuable insight into the emotion of text?
    
  
    
### February 1st 
* summary of what has ben done so far
* k fold cross validation

### February 7th
* using 2000 random seed, bigram 70,000 features
* stratified shuffle split
* 87.71 -> 87.41?????? 
* this is taking the bigram with 70,000 features, which performed best for the random seed of 2000 and using 5 fold cross validation, the split is entirely different here!
* those values are not comprable
* lets revisit that tomorrow! 

### February 9th 
* Experimental setup
* maybe spin up a quick web page?
* Finding a good evaluation metric 
65
### Feb 12th
* Experimental Design
* Which comparison metric? How is acc() calculated?
* Validation sets
* Calculate F1 Score?

### Feb 15th
* Changing number of features to see if trigram levels out
* Put old stuff into F1 scores, features selection, whats that about?? 

### Feb 21st 
* lets sort out this with new models, just do a graph for each! 
* tried the tdidf vectoriser, made like no difference at all 
* try with the different models, do with set value for n and just trigram else it will take too much time
* KNN for the graph
* lets get them bae(yes)
*  next time move onto getting the graph a bit more sorted out, move the axes??
* write up the basis for overall report

### Feb 22nd 
* do KNN for the multiple models

### Feb 25th
* was actually graphing the precision before accidentally instead of the F1 score!
* this explains why the values were inconsistent with the varying number of features graph
* values are all much closer now, apart from the values for the nearest centroid, which are wildly off!
* moving on from this I'm going to see if I can get this all together a bit more, relate this to ekmans or whatever
* got lots of stuff to do!
* make a plan for the next 2 weeks yo

### Feb 27th 
* bit of a divergence from the rest of it but im going to get this web server with the API done, want to get it out of the way.
* what I want from this python app is train model - model is chill - input text - output a vad
* a & d are going to be predicted myself with bayes
* lets get those values sorted

### March 5th
* lets get input output sorted 
* unsure how to use the prediction 
* lets get this experimental stuff written up

### March 12th
* for multiclass will need different algorthms, as valence trains over 5 classes, arousal 4 and dominance 3

### March 13th 
* Over/Under sampling data

models wanted to run for oversampling: 
* Random Over sampling
* SMOTE
* Adaptive Synthetic

## 
* k fold statistical analysis
* input output
* Problems
* acheivements