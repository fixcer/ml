import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, homogeneity_score


def preprocess(sentence):
	sentence = sentence.lower()
	stemmer = SnowballStemmer("english")
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(sentence)
	filtered_words = [stemmer.stem(w) for w in tokens]

	return " ".join(filtered_words)

data = pd.read_csv('./dataset.csv')
x = data['text'].tolist()

for index, value in enumerate(x):
    x[index] = preprocess(x[index])

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(x)


true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', n_init=1)
model.fit(X)
prediction = model.predict(X)
print("Accuracy:", '{:.2f}'.format(homogeneity_score(data.category, prediction)))

pca = PCA(n_components=2, random_state=0)
reduced_features = pca.fit_transform(X.toarray())
reduced_cluster_centers = pca.transform(model.cluster_centers_)
plt.scatter(reduced_features[:,0], reduced_features[:,1], c=prediction)
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')

print("\nTop terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :15]:
        print('%s' % terms[ind], end=" ")
    print('\n')

print("Prediction")

y_test = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(y_test)
print(prediction)

plt.show()