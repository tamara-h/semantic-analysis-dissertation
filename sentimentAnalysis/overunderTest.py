import pandas
import numpy as np

emoBank = pandas.read_csv("./data/emoBank2.csv", sep="\t")
emoBankBinary = pandas.read_csv("./data/emoBank2.csv", sep="\t")

x = emoBank.sentence
y = emoBank.Valence

nullacc = 0

def sample(ngram_range=(1,2), n_features=85000, methods=[]):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import f1_score
    from imblearn.pipeline import make_pipeline

    methodsResults = pandas.DataFrame()
    counter = -2
    for osm in methods:
        counter += 1
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=3000)
        # confusion_sum = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        result = []
        for train_index, test_index in sss.split(x, y):
            cvec = CountVectorizer()
            cvec.set_params(max_features=n_features, ngram_range=ngram_range)
            clf = MultinomialNB()

            if(osm == 0):
                pipeline = make_pipeline(cvec, clf)
            else:
                pipeline = make_pipeline(cvec, osm, clf)


            # X = cvec.fit_transform(x)
            x_train, x_test = x[train_index], x[test_index]
            # X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            sentiment_fit = pipeline.fit(x_train, y_train)
            y_pred = sentiment_fit.predict(x_test)

            # conmat = np.array(confusion_matrix(y_test, y_pred, labels=[2.0, 3.0, 4.0]))
            # print(conmat)
            f1 = f1_score(y_test, y_pred, labels=[2.0, 3.0, 4.0], average="micro")
            result.append(f1)
        # print(result)
        if(osm == 0):
            methodsResults["Base Case"] = result
        else:
            methodsResults[type(osm).__name__] = result
        # print(confusion_sum)
    return methodsResults


def plotRes(classifiersDataframe, isOversampling):
    import matplotlib.pyplot as plt

    columnNames = []
    barValues = []

    for column in classifiersDataframe:

        columnNames.append(column)
        columnValue = 0
        count = 0
        for row in classifiersDataframe[column]:
            columnValue += row
            count += 1
        averageColScore = columnValue / count
        barValues.append(averageColScore)

    n_groups = len(columnNames)
    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    opacity = 0.6



    if(isOversampling):
        ax.bar(index, barValues,
               alpha=opacity, color='g',
               label='bigram with 85,000 features')
        ax.set_xlabel('Oversampling Method')
        ax.set_title('F1 score by oversampling method')
        ax.set_ylim([0.6, 0.8])
    else:
        ax.bar(index, barValues,
               alpha=opacity, color='c',
               label='bigram with 85,000 features')
        ax.set_xlabel('Undersampling Method')
        ax.set_title('F1 score by undersampling method')
        ax.set_ylim([0.2, 0.8])

    ax.set_ylabel('F1 Scores')


    ax.set_xticks(index)
    ax.set_xticklabels(columnNames)
    ax.legend()

    fig.tight_layout()
    plt.show()



import time
start_time = time.time()

def main():
    SEED = 777

    from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
    from imblearn.under_sampling import NearMiss, RandomUnderSampler

    runOverSample = False

    if runOverSample:
        ROS = RandomOverSampler(random_state=SEED)
        SMOTE = SMOTE(random_state=SEED)
        ADASYN = ADASYN(ratio='minority', random_state=SEED)
        results = sample(methods=[0, ROS, SMOTE, ADASYN])
        results_dataset = pandas.DataFrame(data=results)
        pathString = "./data/oversample.csv"
        results_dataset.to_csv(pathString, sep="\t")

    else:

        RUS = RandomUnderSampler(random_state=SEED)
        nearMiss1 = NearMiss(ratio='not minority', random_state=SEED, version=1)

        results = sample(methods=[0, RUS, nearMiss1])

        results_dataset = pandas.DataFrame(data=results)
        pathString = "./data/undersample.csv"
        results_dataset.to_csv(pathString, sep="\t")

    plotRes(results, runOverSample)

main()