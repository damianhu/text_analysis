import tensorflow as tf
import keras as kr

import re
import nltk.tokenize
import gensim
from sklearn.model_selection import train_test_split
from keras.layers import Embedding
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.core import Flatten
import numpy as np
pattern1 = re.compile(r'!|\*|#|\$|%|&|\(|\)|"|\+|/|:|;|<|=|>|@|\[|\\|\]|\^|\`|\{|\||\}|\~|\t|\n')


def read_data(path):
    with open(path, "r") as f:
        result = []
        for line in f:
            line = re.sub(pattern1, '', line)
            line = line.lower()
            line = line.replace(",", '')
            line = line.replace(".", '')
            line = line.replace("?", '')
            result.append(line)
        return result


result1 = read_data("neg.txt")
result2 = read_data("pos.txt")
vecmodel = gensim.models.Word2Vec.load('vecmodel')
X = result1+result2
y = [0]*len(result1) + [1]*len(result2)
t = Tokenizer(num_words=5000)
y = to_categorical(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

t.fit_on_texts(X_train)
max_length = 4
X_train = t.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=max_length, padding='post')

X_test = t.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_length, padding='post')

vocab_size = len(t.word_index) + 1

X_val = t.texts_to_sequences(X_val)
X_val = pad_sequences(X_val, maxlen=max_length, padding='post')

init = kr.initializers.glorot_uniform(seed=1)


word2idx = {'_PAD': 0}

vocab_list = [(k, vecmodel.wv[k]) for k, v in vecmodel.wv.vocab.items()]
embeddings_matrix = np.zeros((len(vecmodel.wv.vocab.items()) + 1, vecmodel.vector_size))
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    word2idx[word] = i+1
    embeddings_matrix[i+1] = vocab_list[i][1]

EMBEDDING_DIM = 100
embedding_layer = Embedding(len(vecmodel.wv.vocab.items()) + 1,
                            EMBEDDING_DIM,
                            weights=[embeddings_matrix],
                            trainable=False,
                            input_length=4
                            )

# init = kr.initializers.glorot_uniform(seed=1)
simple_adam = kr.optimizers.Adam()
model = kr.models.Sequential()
model.add(embedding_layer)
model.add(Flatten())
model.add(kr.layers.Dense(units=100, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(kr.layers.Dense(units=2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])

b_size = 100
max_epochs = 10
print("Starting training ")
print("xtrain shape:"+str(np.array(X_train).shape))
print("ytrain shape:"+str(np.array(y_train).shape))

h = model.fit(np.array(X_train), np.array(y_train), batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1, validation_data=(X_val, y_val),)
print("Training finished \n")
eval = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" % (eval[0], eval[1] * 100))


# kernel_regularizer=regularizers.l2(0.01)



