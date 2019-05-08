import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def wordAnalysis():
    def wordCloud():
        emoBank = pandas.read_csv("./data/emoBank2.csv", sep="\t")

        pos_sentences = emoBank[emoBank.Valence < 1.5]
        print(len(pos_sentences.index))
        pos_string = []
        for t in pos_sentences.sentence:
            pos_string.append(t)
        pos_string = pandas.Series(pos_string).str.cat(sep=' ')

        from wordcloud import WordCloud

        wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(pos_string)
        plt.figure(figsize=(12, 10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def wordFrequencies(emoBank):

        cvec = CountVectorizer()
        cvec.fit(emoBank.sentence)
        numberOfWords = len(cvec.get_feature_names())

        neg_doc_matrix = cvec.transform(emoBank[emoBank.Valence <= 2.5].sentence)
        pos_doc_matrix = cvec.transform(emoBank[emoBank.Valence > 2.5].sentence)
        neg_tf = np.sum(neg_doc_matrix, axis=0)
        pos_tf = np.sum(pos_doc_matrix, axis=0)
        neg = np.squeeze(np.asarray(neg_tf))
        pos = np.squeeze(np.asarray(pos_tf))
        # print(cvec.get_feature_names())
        term_freq_df = pandas.DataFrame([neg, pos], columns=cvec.get_feature_names()).transpose()
        term_freq_df.columns = ['negative', 'positive']
        term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']

        # print(term_freq_df.head(10))
        # print(term_freq_df.sort_values(by='total', ascending=False).iloc[:10])
        return(term_freq_df)


        #
        # term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
        # term_freq_df.sort_values(by='total', ascending=False).iloc[:10]
        # print(term_freq_df.head())

    def removeStopWords(my_df):
        from nltk.corpus import stopwords
        stoplist = set(stopwords.words('english'))
        new_df = my_df
        for w in my_df.index.values:
            if w in stoplist:
                new_df.drop(w, inplace=True)
        return new_df

    def harmonicMeans(df):
        # See how frequent the word occours in pos or neg
        # This is an issue due to the fact that most values are in the positive rate
        df['pos_rate'] = df['positive']/df['total']
        df['neg_rate'] = df['negative'] * 1. / df['total']
        # Frequency a word occours in a class
        # Also issue due to class imbalance
        df['pos_freq_pct'] = df['positive'] * 1./df['positive'].sum()
        df['neg_freq_pct'] = df['negative'] * 1. / df['negative'].sum()
        # Using harmonic mean
        from scipy.stats import hmean
        df['pos_hmean'] = df.apply(lambda x: (hmean([x['pos_rate'], x['pos_freq_pct']])
                                    if x['pos_rate'] > 0 and x['pos_freq_pct'] > 0 else 0), axis=1)

        df['neg_hmean'] = df.apply(
            lambda x: (hmean([x['neg_rate'], x['neg_freq_pct']]) if x['neg_rate'] > 0 and x['neg_freq_pct'] > 0 else 0),
            axis=1)

        from scipy.stats import norm

        def normcdf(x):
            return norm.cdf(x, x.mean(), x.std())

        df['pos_rate_normcdf'] = normcdf(df['pos_rate'])
        df['pos_freq_pct_normcfd'] = normcdf(df['pos_freq_pct'])
        df['pos_normcdf_hmean'] = normcdf(hmean([df['pos_rate_normcdf'], df['pos_freq_pct_normcfd']]))

        df['neg_rate_normcdf'] = normcdf(df['neg_rate'])
        df['neg_freq_pct_normcdf'] = normcdf(df['neg_freq_pct'])
        df['neg_normcdf_hmean'] = hmean(
            [df['neg_rate_normcdf'], df['neg_freq_pct_normcdf']])

        # print(df.sort_values(by='neg_normcdf_hmean', ascending=False).iloc[:10])

        import seaborn as sns

        plt.figure(figsize=(8, 6))
        ax = sns.regplot(x="neg_hmean", y="pos_hmean", fit_reg=False, scatter_kws={'alpha': 0.5}, data=df)
        plt.ylabel('Positive Rate and Frequency Harmonic Mean')
        plt.xlabel('Negative Rate and Frequency Harmonic Mean')
        plt.title('neg_hmean vs pos_hmean')
        # plt.show()

        plt.figure(figsize=(8, 6))
        ax = sns.regplot(x="neg_normcdf_hmean", y="pos_normcdf_hmean", fit_reg=False, scatter_kws={'alpha': 0.5},
                         data=df)
        plt.ylabel('Positive Rate and Frequency CDF Harmonic Mean')
        plt.xlabel('Negative Rate and Frequency CDF Harmonic Mean')
        plt.title('neg_normcdf_hmean vs pos_normcdf_hmean')

        # plt.show()

        from bokeh.plotting import figure
        from bokeh.io import output_file, show
        from bokeh.models import LinearColorMapper
        from bokeh.models import HoverTool

        output_file('inc_stopwords.html')
        color_mapper = LinearColorMapper(palette='Inferno256', low=min(df.pos_normcdf_hmean),
                                         high=max(df.pos_normcdf_hmean))

        p = figure(x_axis_label='neg_normcdf_hmean', y_axis_label='pos_normcdf_hmean')

        p.circle('neg_normcdf_hmean', 'pos_normcdf_hmean', size=5, alpha=0.3, source=df,
                 color={'field': 'pos_normcdf_hmean', 'transform': color_mapper})

        hover = HoverTool(tooltips=[('token', '@index')])
        p.add_tools(hover)
        return df
        # show(p)
    #
    # emoBank = pandas.read_csv("./data/emoBank.csv", sep="\t")
    # # Get new dataframe
    # df = wordFrequencies(emoBank)
    # return harmonicMeans(removeStopWords(df))

def generateEmoBank():
    import string
    textRaw = pandas.read_csv("./data/raw.tsv", sep='\t', quoting=3)
    readerValues = pandas.read_csv("./data/reader.tsv", sep='\t')

    textRaw.set_index("id").join(readerValues.set_index("id"), on="id")

    result = pandas.merge(textRaw.set_index("id"),
                          readerValues.set_index("id"),
                          on="id")

    sentencesDf = pandas.DataFrame(data=result['sentence'])
    result.drop(columns=["sd.Arousal", "sd.Dominance", "sd.Valence", "freq", "sentence"], inplace=True)
    # result.drop(columns=["sd.Arousal", "sd.Dominance", "sd.Valence", "freq"], inplace=True)



    print(sentencesDf.columns.values)
    print(sentencesDf.head())


    exclude = set(string.punctuation)
    for index, row in sentencesDf.iterrows():
        result.at[index, 'sentence'] = ''.join(ch for ch in row['sentence'] if ch not in exclude).encode(encoding='UTF-8',errors='strict')



    result['Valence'] = round(result['Valence'])
    result['Arousal'] = round(result["Arousal"])
    result['Dominance'] = round(result["Dominance"])

    result['sentence'] = np.where(result["sentence"] == None, 0, result["sentence"])
    # print(result['sentence'])


    # result['sentence'] = ''.join(ch for ch in result['sentence'] if ch not in exclude)

    # result['Valence'] = np.where(result["Valence"] == 1, 2, result["Valence"])
    # result['Arousal'] = np.where(result["Arousal"] == 1, 2, result["Arousal"])
    # result['Dominance'] = np.where(result["Dominance"] == 1, 2, result["Dominance"])
    #
    # result['Valence'] = np.where(result["Valence"] == 5 , 4, result["Valence"])
    # result['Arousal'] = np.where(result["Arousal"] == 5 , 4, result["Arousal"])
    # result['Dominance'] = np.where(result["Dominance"] == 5 , 4, result["Dominance"])


    result.to_csv("./data/emoBank3.csv", sep="\t")


def analyseEmoBankBinary():
    emoBank = pandas.read_csv("./data/emoBank.csv", sep="\t")

    pos_vad = 0
    pos_va_neg_d = 0
    pos_v_neg_ad = 0
    pos_ad_neg_v = 0
    pos_a_neg_vd = 0
    pos_d_neg_av = 0
    pos_vd_neg_a = 0
    neg_vad = 0

    for index, row in emoBank.iterrows():
        found = False

        if row['Valence'] == 0:
            if row['Dominance'] == 0:
                if row['Arousal'] == 0:
                    neg_vad += 1
                else:
                    pos_a_neg_vd += 1
            else:
                if row['Arousal'] == 0:
                    pos_d_neg_av += 1
                else:
                    pos_ad_neg_v += 1
        else:
            if row['Dominance'] == 0:
                if row['Arousal'] == 0:
                    pos_v_neg_ad += 1
                else:
                    pos_va_neg_d += 1
            else:
                if row['Arousal'] == 0:
                    pos_vd_neg_a += 1
                else:
                    pos_vad += 1

    print('pos_vad: ' + str(pos_vad))
    print('pos_vd_neg_a: ' + str(pos_vd_neg_a))
    print('pos_va_neg_d: ' + str(pos_va_neg_d))
    print('pos_v_neg_ad: ' + str(pos_v_neg_ad))
    print('pos_ad_neg_v: ' + str(pos_ad_neg_v))
    print('pos_d_neg_av: ' + str(pos_d_neg_av))
    print('pos_a_neg_vd: ' + str(pos_a_neg_vd))
    print('neg_vad: ' + str(neg_vad))


def analyseEmoBankNonBinary():
    # emoBank = pandas.read_csv("./data/emoBank.csv", sep="\t")
    emoBank = pandas.read_csv("./data/emoBank3.csv", sep="\t")
    #
    print(emoBank.Valence.value_counts())
    print(emoBank.Dominance.value_counts())
    print(emoBank.Arousal.value_counts())

    from statsmodels.graphics.mosaicplot import mosaic

    # mosaic(emoBank, ["Valence", "Arousal", "Dominance"], axes_label=True, labelizer= lambda k: "")
    plt.show()


def roundLexiconDataset():
    lexicon = pandas.read_csv("./data/word_edited_raw.csv", sep=",")

    lexicon['v'] = round(lexicon['v'], 1)
    lexicon['a'] = round(lexicon['a'], 1)
    lexicon['d'] = round(lexicon['d'], 1)

    lexicon.to_csv("./data/lexicon.csv", sep="\t")


# analyseEmoBankNonBinary()
# generateEmoBank()

roundLexiconDataset()