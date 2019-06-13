import re
import nltk.tokenize
import gensim
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
            result.append(list(line.strip('\n').split(" ")))
        return result


result1 = read_data("neg.txt")
result2 = read_data("pos.txt")

res = result1+result2

model = gensim.models.Word2Vec(res)
con = model.most_similar(positive=['good'], negative=['bad'], topn=20)
fileobject = open('poscon.txt', 'w')
for ip in con:
    fileobject.write(str(ip))
    fileobject.write('\n')
fileobject.close()


print(con)
