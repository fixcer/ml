import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


STOPWORDS = set(stopwords.words('english'))

def preprocess(sentence):
	sentence = sentence.lower()
	stemmer = SnowballStemmer("english")
	lemmatizer = WordNetLemmatizer()
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(sentence)
	tokens = [stemmer.stem(w) for w in tokens if w not in STOPWORDS]
	filtered_words = [lemmatizer.lemmatize(token) for token in tokens]

	return " ".join(filtered_words)

df = pd.read_csv('../dataset/dataset.csv')
df = df[pd.notnull(df['category'])]
print(df.head())

my_tags = ['politics','tech']
plt.figure()
df.category.value_counts().plot(kind='bar');
plt.show()

print("\nBefore:", df['text'].apply(lambda x: len(x.split(' '))).sum())
df['text'] = df['text'].apply(preprocess)
print("After:", df['text'].apply(lambda x: len(x.split(' '))).sum())

X = df.text
y = df.category
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
plt.figure()
y_test.value_counts().plot(kind='bar');
plt.show()

print("\nTrain size:", X_train.shape[0])
print("Test size:", X_test.shape[0])


model = Pipeline([('vect', CountVectorizer(max_features=3000)),
					('tfidf', TfidfTransformer()),
					('clf', LinearSVC()),])
					# ('clf', MultinomialNB()),])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('\nAccuracy %s' % accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred,target_names=my_tags))