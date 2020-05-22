import numpy as np
import pandas as pd
import nltk
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def preprocess(sentence):
	sentence = sentence.lower()
	stemmer = SnowballStemmer("english")
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(sentence)
	filtered_words = [stemmer.stem(w) for w in tokens]

	return " ".join(filtered_words)

data = pd.read_csv('../dataset/dataset.csv')
x = data['text'].tolist()
y = data['category'].tolist()

for index, value in enumerate(x):
    print("Processing data:", index+1)
    x[index] = preprocess(x[index])

count_vect = CountVectorizer()
X = count_vect.fit_transform(x)
pickle.dump(count_vect.vocabulary_, open("count_vector.pkl","wb"))

tfidf_transformer = TfidfTransformer()
X = tfidf_transformer.fit_transform(X)
pickle.dump(tfidf_transformer, open("tfidf.pkl","wb"))
Y = np.array(y)

print("\nNo of features extracted:",X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


model = MultinomialNB().fit(X_train, y_train)
pickle.dump(model, open("nb_model.pkl", "wb"))

y_pred = model.predict(X_test)

c_mat = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)

print("Confusion Matrix:\n", c_mat)
print("\nAccuracy:", '{:.2f}'.format(acc))
# print("\nPrecision:", '{:.2f}'.format(precision))
# print("\nRecall:", '{:.2f}'.format(recall))