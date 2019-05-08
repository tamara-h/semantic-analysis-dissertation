import numpy as np
import pandas
emoBank = pandas.read_csv("./data/emoBank.csv", sep="\t")

x = emoBank.sentence
y = emoBank.Valence

# Test out stratified split with a fixed bigram and number of features
def stratifiedSplitFixed():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import Pipeline
    from scipy import stats

    x = emoBank.sentence
    y = emoBank.Valence

    result = []
    myTest = 0
    count = 0

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=3000)
    confusion_sum = [[0,0],[0,0]]

    for train_index, test_index in sss.split(x, y):
        cvec = CountVectorizer()
        cvec.set_params(stop_words=None, max_features=200000, ngram_range=(1,3))
        # X = cvec.fit_transform(x)
        x_train, x_test = x[train_index], x[test_index]
        # X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = LogisticRegression(solver='liblinear')

        pipeline = Pipeline([
            ('vectorizer', cvec),
            ('classifier', clf)
        ])

        sentiment_fit = pipeline.fit(x_train, y_train)
        predictions = sentiment_fit.predict(x_test)

        newX = pandas.DataFrame({"Constant": np.ones(len(x_test))}).join(pandas.DataFrame(x_test))
        # print(y_test - predictions)
        # print(sum((y_test - predictions) ** 2))
        # print((len(newX) - len(newX.columns)))
        MSE = (sum((y_test - predictions) ** 2)) / (len(newX) - len(newX.columns))

        print(MSE)
        print(newX)
        print(np.dot(newX.T, newX))
        print(np.linalg.inv(np.dot(newX.T, newX)).diagonal())
        var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = params / sd_b

        p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

        sd_b = np.round(sd_b, 3)
        ts_b = np.round(ts_b, 3)
        p_values = np.round(p_values, 3)
        params = np.round(params, 4)

        myDF3 = pandas.DataFrame()
        myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilites"] = [params, sd_b, ts_b,
                                                                                                     p_values]
        print(myDF3)

    return sentiment_fit




def train():
    model = stratifiedSplitFixed()
    test_array = np.empty((2065,))
    test_array = ["" for x in range(test_array.size)]
    test_array[0] = "Im so sad this is actually very upsetting I hate my life I am full of death and being sad and depressed"
    test_array[1] = "I love life im so happy and im so glad that I exist in this beautiful world"
    test_array[2] = "so far my week has been okay, I dont really know what im doing and im a little stressed due to the amount of things going on. Ah well."

    sad = 0
    happy = 0
    neutral = 0
    ITERATIONS = 1
    for i in range(ITERATIONS):
        prediction = model.predict(test_array)
        print(prediction)
        sad = sad + prediction[0]
        happy = happy + prediction[1]
        neutral = neutral + prediction[2]
    print("sad")
    print(sad/ITERATIONS)
    print("happy")
    print(happy/ITERATIONS)
    print("neutral")
    print(neutral/ITERATIONS)

train()