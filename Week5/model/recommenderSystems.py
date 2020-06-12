import numpy as np
import pandas as pd
import pickle
import nltk
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Chưa ưu tiên bài báo đọc gần nhất

def last(n): 
    return n[0]   


def sort(tuples):  
    return sorted(tuples, key = last)


def preprocess(sentence):
	sentence = sentence.lower()
	stemmer = SnowballStemmer("english")
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(sentence)
	filtered_words = [stemmer.stem(w) for w in tokens]

	return " ".join(filtered_words)


data = pd.read_csv('../dataset/seen.csv')
x = data['text'].tolist()

for index, value in enumerate(x):
    x[index] = preprocess(x[index])

loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
loaded_model = pickle.load(open("nb_model.pkl","rb"))

X = loaded_vec.transform(x)
X = loaded_tfidf.transform(X)
predicted = loaded_model.predict(X)

most_seen = Counter(predicted)
tmp = 0
for k, v in most_seen.items():
	tmp = max(tmp, v)

for k, v in most_seen.items():
	if v == tmp:
		most_seen = k
		break
print(most_seen + '\n' + "-"*100)

data = pd.read_csv('../dataset/dataset.csv')
data = data.loc[data['category'] == most_seen]
x = data["text"].sample(frac=1).reset_index(drop=True).tolist()
for i in x[:10]:
	print(i, "\n", "-"*100)
