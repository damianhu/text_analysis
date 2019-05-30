
import re
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import sys

pattern1 = re.compile(r'!|\*|#|\$|%|&|\(|\)|"|\+|/|:|;|<|=|>|@|\[|\\|\]|\^|\`|\{|\||\}|\~|\t|\n')
stop_words = set(stopwords.words('english'))
def read_data(path):
    result = []
    with open(path, 'r') as f:
        for line in f:
            line = re.sub(pattern1, '', line)
            line = line.replace(",", ' ,')
            line = line.replace(".", ' .')
            result.append(list(line.strip('\n').split(" ")))
    return result



def stop_word(reader):
    result = []
    for item in reader:
        new = []
        for i in item:
            if i.lower() not in stop_words:
                new.append(i)
        result.append(new)
    return result


def split_data(data, label):
    X = data
    y = [label]*len(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=1)
    return X_train, X_test, X_val




if __name__ == "__main__":
    input_path = sys.argv[1]
    data = read_data(input_path)
    no_stop_data = stop_word(data)
    if input_path == "pos.txt":
        train_list, test_list, val_list = split_data(data, 1)
        train_list_no_stopword, test_list_no_stopword, val_list_no_stopword = split_data(no_stop_data, 1)
    else:
        train_list, test_list, val_list = split_data(data, 0)
        train_list_no_stopword, test_list_no_stopword, val_list_no_stopword = split_data(no_stop_data, 0)


    """
    Tokenize the input file here
    Create train, val, and test sets
    """

    # sample_tokenized_list = [["Hello", "World", "."], ["Good", "bye"]]

    np.savetxt("train.csv", train_list, delimiter=",", fmt='%s')
    np.savetxt("val.csv", val_list, delimiter=",", fmt='%s')
    np.savetxt("test.csv", test_list, delimiter=",", fmt='%s')

    np.savetxt("train_no_stopword.csv", train_list_no_stopword,
               delimiter=",", fmt='%s')
    np.savetxt("val_no_stopword.csv", val_list_no_stopword,
               delimiter=",", fmt='%s')
    np.savetxt("test_no_stopword.csv", test_list_no_stopword,
               delimiter=",", fmt='%s')




# print(no_stop_neg)













