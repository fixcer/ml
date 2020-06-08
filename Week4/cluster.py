import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, homogeneity_score

STOPWORDS = set(stopwords.words('english'))

def preprocess(sentence):
	sentence = sentence.lower()
	stemmer = SnowballStemmer("english")
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(sentence)
	filtered_words = [stemmer.stem(w) for w in tokens if w not in STOPWORDS]

	return " ".join(filtered_words)


def raw():
	pca = PCA(n_components=2, random_state=42)
	reduced_features = pca.fit_transform(X.toarray())
	reduced_cluster_centers = pca.transform(model.cluster_centers_)
	plt.scatter(reduced_features[:,0], reduced_features[:,1], c=prediction)
	plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')

	plt.show()


def result():
	print("\nTop terms per cluster:")
	order_centroids = model.cluster_centers_.argsort()[:, ::-1]
	terms = vectorizer.get_feature_names()
	for i in range(true_k):
		print("Cluster %d:" % i),
		for ind in order_centroids[i, :15]:
			print('%s' % terms[ind], end=" ")
		print('\n')


def test():
	print("Prediction Test")
	y_test = vectorizer.transform([input()])
	prediction = model.predict(y_test)
	print(prediction)


df = pd.read_csv('./dataset.csv')
df['text'] = df['text'].apply(preprocess)
vectorizer = TfidfVectorizer(min_df=3, max_features=3000) # Có sử dụng idf theo mặc định
X = vectorizer.fit_transform(df['text'])

true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=3000, tol=1e-4) # Khoảng cách euclide
model.fit(X)
# print(model.cluster_centers_)
prediction = model.predict(X)
print("Accuracy:", '{:.2f}'.format(homogeneity_score(df.category, prediction)))

result()
raw()
test()
