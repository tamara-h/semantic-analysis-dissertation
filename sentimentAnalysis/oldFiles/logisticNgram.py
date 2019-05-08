import pandas
import numpy as np

# This has been taken from the towards data science sentiment analysis information.

# processed_words = dataProcess.wordAnalysis()
emoBank = pandas.read_csv("./data/emoBank2.csv", sep="\t")

x = emoBank.sentence
y = emoBank.Valence


from sklearn.model_selection import train_test_split, cross_val_score
SEED = 2000

nullacc = 0

x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.35, random_state=SEED)

x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)



print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train),
                                                                             (len(x_train[y_train == 0]) / (len(x_train)*1.))*100,
                                                                            (len(x_train[y_train == 1]) / (len(x_train)*1.))*100))
print("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation),
                                                                             (len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100,
                                                                            (len(x_validation[y_validation == 1]) / (len(x_validation)*1.))*100))
print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test),
                                                                             (len(x_test[y_test == 0]) / (len(x_test)*1.))*100,
                                                                            (len(x_test[y_test == 1]) / (len(x_test)*1.))*100))


print("******************************")



def countVectorize():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight')
    from time import time

    def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
        if len(x_test[y_test == 0]) / (len(x_test) * 1.) > 0.5:
            null_accuracy = len(x_test[y_test == 0]) / (len(x_test) * 1.)
        else:
            null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test) * 1.))
        global nullacc
        nullacc = null_accuracy
        t0 = time()
        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        train_test_time = time() - t0
        accuracy = accuracy_score(y_test, y_pred)
        print("null accuracy: {0:.2f}%".format(null_accuracy * 100))
        print("accuracy score: {0:.2f}%".format(accuracy * 100))
        if accuracy > null_accuracy:
            print("model is {0:.2f}% more accurate than null accuracy".format((accuracy - null_accuracy) * 100))
        elif accuracy == null_accuracy:
            print("model has the same accuracy with the null accuracy")
        else:
            print("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy - accuracy) * 100))
        print("train and test time: {0:.2f}s".format(train_test_time))
        print("-" * 80)
        return accuracy, train_test_time

    cvec = CountVectorizer()
    lr = LogisticRegression()
    n_features = np.arange(10000, 100001, 10000)


    def nfeature_accuracy_checker(vectorizer=cvec, n_features=n_features, stop_words=None, ngram_range=(1, 1),
                                  classifier=lr):
        result = []
        print(classifier)
        print("\n")
        for n in n_features:
            vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
            checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
            print("Validation result for {} features".format(n))
            nfeature_accuracy, tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation,
                                                          y_validation)
            result.append((n, nfeature_accuracy, tt_time))
        return result



    print("RESULT FOR UNIGRAM WITHOUT STOP WORDS\n")
    feature_result_wosw = nfeature_accuracy_checker(stop_words='english')

    print("RESULT FOR UNIGRAM WITH STOP WORDS\n")
    feature_result_ug = nfeature_accuracy_checker()

    print("RESULT FOR BIGRAM WITH STOP WORDS\n")
    feature_result_bg = nfeature_accuracy_checker(ngram_range=(1, 2))

    print("RESULT FOR TRIGRAM WITH STOP WORDS\n")
    feature_result_tg = nfeature_accuracy_checker(ngram_range=(1, 3))

    nfeatures_plot_tg = pandas.DataFrame(feature_result_tg,
                                         columns=['nfeatures', 'validation_accuracy', 'train_test_time'])
    nfeatures_plot_bg = pandas.DataFrame(feature_result_bg,
                                         columns=['nfeatures', 'validation_accuracy', 'train_test_time'])
    nfeatures_plot_ug = pandas.DataFrame(feature_result_ug,
                                         columns=['nfeatures', 'validation_accuracy', 'train_test_time'])

    plt.figure(figsize=(8, 6))
    plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy, label='trigram')
    plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy, label='bigram')
    plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram')
    # plt.plot([10000,100000], [nullacc, nullacc], label='zero accuracy')
    plt.title("N-gram(1~3) test result : Accuracy")
    plt.xlabel("Number of features")
    plt.ylabel("Validation set accuracy")
    plt.legend()
    plt.show()

    def train_test_and_evaluate(pipeline, x_train, y_train, x_test, y_test):
        if len(x_test[y_test == 0]) / (len(x_test) * 1.) > 0.5:
            null_accuracy = len(x_test[y_test == 0]) / (len(x_test) * 1.)
        else:
            null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test) * 1.))
        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0, 1]))
        confusion = pandas.DataFrame(conmat, index=['negative', 'positive'],
                                     columns=['predicted_negative', 'predicted_positive'])
        print("null accuracy: {0:.2f}%".format(null_accuracy * 100))
        print("accuracy score: {0:.2f}%".format(accuracy * 100))
        if accuracy > null_accuracy:
            print("model is {0:.2f}% more accurate than null accuracy".format((accuracy - null_accuracy) * 100))
        elif accuracy == null_accuracy:
            print("model has the same accuracy with the null accuracy")
        else:
            print("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy - accuracy) * 100))
        print("-" * 80)
        print("Confusion Matrix\n")
        print(confusion)
        print("-" * 80)
        print("Classification Report\n")
        print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

    tg_cvec = CountVectorizer(max_features=80000, ngram_range=(1, 3))
    tg_pipeline = Pipeline([
        ('vectorizer', tg_cvec),
        ('classifier', lr)
    ])
    train_test_and_evaluate(tg_pipeline, x_train, y_train, x_validation, y_validation)






countVectorize()