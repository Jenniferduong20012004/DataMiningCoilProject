import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
model = None
vectorizer = None
import joblib
model = joblib.load(r'F:\DataMiningCoilProject\code\LOGISTIC_REGRESSION.pkl')

X_test = ["I completely curious"]
with open(r'F:\DataMiningCoilProject\code\tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

X_tfidf = vectorizer.transform(X_test)
y_pred = model.predict(X_tfidf)
print("Predicted class:", y_pred)

