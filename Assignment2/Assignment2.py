from sklearn.naive_bayes import MultinomialNB
import nltk
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import sys

import numpy as np

positive = 1
negative = 0


def read_data(path):
    res = []
    with open(path, 'r') as f:

        for line in f:
            res.append(line)
    return res


neg_train_list = read_data("neg/train.csv")
neg_test_list = read_data("neg/test.csv")
neg_val_list = read_data("neg/val.csv")
neg_train_no_stop_list = read_data("neg/train_no_stopword.csv")
neg_test_no_stop_list = read_data("neg/test_no_stopword.csv")
neg_val_no_stop_list = read_data("neg/val_no_stopword.csv")
pos_train_list = read_data("pos/train.csv")
pos_test_list = read_data("pos/test.csv")
pos_val_list = read_data("pos/val.csv")
pos_train_no_stop_list = read_data("pos/train_no_stopword.csv")
pos_test_no_stop_list = read_data("pos/test_no_stopword.csv")
pos_val_no_stop_list = read_data("pos/val_no_stopword.csv")


def stop_words():

    X_train_list = neg_train_list + pos_train_list
    y_train_stopwords = [negative]*len(neg_train_list) + [positive]*len(pos_train_list)
    X_test_list = neg_test_list + pos_test_list
    y_test_stopwords = [negative]*len(neg_test_list) + [positive]*len(pos_test_list)
    X_val_list = neg_val_list + pos_val_list
    y_val_stopwords = [negative]*len(neg_val_list) + [positive]*len(pos_val_list)
    vectorizer = TfidfVectorizer()
    bi_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    ubi_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_list)
    bi_X_train = bi_vectorizer.fit_transform(X_train_list)
    ubi_X_train = ubi_vectorizer.fit_transform(X_train_list)
    X_test = vectorizer.transform(X_test_list)
    bi_X_test = bi_vectorizer.transform(X_test_list)
    ubi_X_test = ubi_vectorizer.transform(X_test_list)
    X_val = vectorizer.transform(X_val_list)
    bi_X_val = bi_vectorizer.transform(X_val_list)
    ubi_X_val = ubi_vectorizer.transform(X_val_list)

    clf = MultinomialNB()
    clf.fit(X_train, y_train_stopwords)

    y_true = clf.predict(X_test)
    clf1 = MultinomialNB(alpha=0.3)
    clf1.fit(bi_X_train, y_train_stopwords)
    y_true2 = clf1.predict(bi_X_test)
    clf2 = MultinomialNB()
    clf2.fit(ubi_X_train, y_train_stopwords)
    y_true3 = clf2.predict(ubi_X_test)


    print("unigram_accuracy: "+str(accuracy_score(y_test_stopwords, y_true)))
    print("bigram_accuracy: "+str(accuracy_score(y_test_stopwords, y_true2)))
    print("ubigram_accuracy: " + str(accuracy_score(y_test_stopwords, y_true3)))


def no_stop_words():

    X_train_list = neg_train_no_stop_list + pos_train_no_stop_list
    y_train_stopwords = [negative] * len(neg_train_no_stop_list) + [positive] * len(pos_train_no_stop_list)
    X_test_list = neg_test_no_stop_list + pos_test_no_stop_list
    y_test_stopwords = [negative] * len(neg_test_no_stop_list) + [positive] * len(pos_test_no_stop_list)
    X_val_list = neg_val_no_stop_list + pos_val_no_stop_list
    y_val_stopwords = [negative] * len(neg_val_no_stop_list) + [positive] * len(pos_val_no_stop_list)
    vectorizer = TfidfVectorizer()
    bi_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    ubi_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_list)
    bi_X_train = bi_vectorizer.fit_transform(X_train_list)
    ubi_X_train = ubi_vectorizer.fit_transform(X_train_list)
    X_test = vectorizer.transform(X_test_list)
    bi_X_test = bi_vectorizer.transform(X_test_list)
    ubi_X_test = ubi_vectorizer.transform(X_test_list)
    X_val = vectorizer.transform(X_val_list)
    bi_X_val = bi_vectorizer.transform(X_val_list)
    ubi_X_val = ubi_vectorizer.transform(X_val_list)

    clf = MultinomialNB()
    clf.fit(X_train, y_train_stopwords)

    y_true = clf.predict(X_test)
    clf1 = MultinomialNB(alpha=0.4)
    clf1.fit(bi_X_train, y_train_stopwords)
    y_true2 = clf1.predict(bi_X_test)
    clf2 = MultinomialNB()
    clf2.fit(ubi_X_train, y_train_stopwords)
    y_true3 = clf2.predict(ubi_X_test)

    print("no_stop_unigram_accuracy: " + str(accuracy_score(y_test_stopwords, y_true)))
    print("no_stop_bigram_accuracy: " + str(accuracy_score(y_test_stopwords, y_true2)))
    print("no_stop_ubigram_accuracy: " + str(accuracy_score(y_test_stopwords, y_true3)))

def no_stop_tune_alpha():

    accuracy_uni = []
    accuracy_bi = []
    accuracy_ubi = []
    alpha = []
    X_train_list = neg_train_no_stop_list + pos_train_no_stop_list
    y_train_stopwords = [negative] * len(neg_train_no_stop_list) + [positive] * len(pos_train_no_stop_list)

    X_val_list = neg_val_no_stop_list + pos_val_no_stop_list
    y_val_stopwords = [negative] * len(neg_val_no_stop_list) + [positive] * len(pos_val_no_stop_list)
    vectorizer = TfidfVectorizer()
    bi_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    ubi_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_list)

    bi_X_train = bi_vectorizer.fit_transform(X_train_list)
    ubi_X_train = ubi_vectorizer.fit_transform(X_train_list)
    X_val = vectorizer.transform(X_val_list)
    bi_X_val = bi_vectorizer.transform(X_val_list)
    ubi_X_val = ubi_vectorizer.transform(X_val_list)
    for a in np.arange(0.1, 2, 0.1):
        clf = MultinomialNB(alpha=a)
        clf.fit(X_train, y_train_stopwords)
        y_true = clf.predict(X_val)
        clf1 = MultinomialNB(alpha=a)
        clf1.fit(bi_X_train, y_train_stopwords)
        y_true2 = clf1.predict(bi_X_val)
        clf2 = MultinomialNB(alpha=a)
        clf2.fit(ubi_X_train, y_train_stopwords)
        y_true3 = clf2.predict(ubi_X_val)
        print("alpha=: "+str(a))
        alpha.append(a)
        print("no_stop_unigram_accuracy: " + str(accuracy_score(y_val_stopwords, y_true)))
        accuracy_uni.append(accuracy_score(y_val_stopwords, y_true))
        print("no_stop_bigram_accuracy: " + str(accuracy_score(y_val_stopwords, y_true2)))
        accuracy_bi.append(accuracy_score(y_val_stopwords, y_true2))
        print("no_stop_ubigram_accuracy: " + str(accuracy_score(y_val_stopwords, y_true3)))
        accuracy_ubi.append(accuracy_score(y_val_stopwords, y_true3))

    plt.xlabel('alpha')
    plt.ylabel('accuracy')
    plt.xticks(np.arange(0.1, 2, .1))
    plt.yticks(np.arange(0.76, 0.83, 0.01))
    plt.plot(alpha, accuracy_uni, label = 'uni_tune')
    plt.plot(alpha, accuracy_bi, label='bi_tune')
    plt.plot(alpha, accuracy_ubi, label='ubi_tune')
    plt.legend()
    plt.savefig('nostop_tunealpha.svg')
    plt.show()




def stop_tune_alpha():
    accuracy_uni = []
    accuracy_bi = []
    accuracy_ubi = []
    alpha = []
    X_train_list = neg_train_list + pos_train_list
    y_train_stopwords = [negative] * len(neg_train_list) + [positive] * len(pos_train_list)

    X_val_list = neg_val_list + pos_val_list
    y_val_stopwords = [negative] * len(neg_val_list) + [positive] * len(pos_val_list)
    vectorizer = TfidfVectorizer()
    bi_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    ubi_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_list)
    bi_X_train = bi_vectorizer.fit_transform(X_train_list)
    ubi_X_train = ubi_vectorizer.fit_transform(X_train_list)
    X_val = vectorizer.transform(X_val_list)
    bi_X_val = bi_vectorizer.transform(X_val_list)
    ubi_X_val = ubi_vectorizer.transform(X_val_list)
    for a in np.arange(0.1, 2, .1):
        clf = MultinomialNB(alpha=a)
        clf.fit(X_train, y_train_stopwords)
        y_true = clf.predict(X_val)
        clf1 = MultinomialNB(alpha=a)
        clf1.fit(bi_X_train, y_train_stopwords)
        y_true2 = clf1.predict(bi_X_val)
        clf2 = MultinomialNB(alpha=a)
        clf2.fit(ubi_X_train, y_train_stopwords)
        y_true3 = clf2.predict(ubi_X_val)
        print("alpha=: " + str(a))
        alpha.append(a)
        print("stop_unigram_accuracy: " + str(accuracy_score(y_val_stopwords, y_true)))
        accuracy_uni.append(accuracy_score(y_val_stopwords, y_true))
        print("stop_bigram_accuracy: " + str(accuracy_score(y_val_stopwords, y_true2)))
        accuracy_bi.append(accuracy_score(y_val_stopwords, y_true2))
        print("stop_ubigram_accuracy: " + str(accuracy_score(y_val_stopwords, y_true3)))
        accuracy_ubi.append(accuracy_score(y_val_stopwords, y_true3))
    plt.xlabel('alpha')
    plt.ylabel('accuracy')
    plt.xticks(np.arange(0.1, 2, .1))
    plt.yticks(np.arange(0.80, 0.85, 0.01))
    plt.plot(alpha, accuracy_uni, label = 'uni_tune')
    plt.plot(alpha, accuracy_bi, label='bi_tune')
    plt.plot(alpha, accuracy_ubi, label='ubi_tune')
    plt.legend()
    plt.savefig('stop_tunealpha.svg')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv)==7:
        input_file1 = sys.argv[1]
        input_file2 = sys.argv[2]
        input_file3 = sys.argv[3]
        input_file4 = sys.argv[4]
        input_file5 = sys.argv[5]
        input_file6 = sys.argv[6]
        if input_file1=="training_pos" and input_file2=="training_neg" and input_file3=="validation_pos" and input_file4=="validation_neg" and input_file5=="test_pos" and input_file6=="test_neg":
            stop_words()
        else:
            no_stop_words()
    else:
        no_stop_words()





# training_pos training_neg validation_pos validation_neg test_pos test_neg













