import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report
import joblib
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "resources", "newPreprocess.csv")
df = pd.read_csv(file_path)
df = df.dropna(subset=['Final_Text'])
# vectorize
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),    
    sublinear_tf=True,     
    max_df=0.9,             
    min_df=5,               
    max_features=50000,    
)
Xfeatures = df['Final_Text']
ylabels = df['label']
X_tfidf = vectorizer.fit_transform(Xfeatures)
sgd_classifier = SGDClassifier(loss='modified_huber', penalty='l2', class_weight='balanced',alpha=0.0001, max_iter=1000, tol=1e-3)
# 10 fold
y_pred = cross_val_predict(sgd_classifier, X_tfidf, ylabels, cv=10)
cm = confusion_matrix(ylabels, y_pred, labels=np.unique(ylabels))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(ylabels))
disp.plot(cmap='Blues')
print("\nClassification Report:")
print(classification_report(ylabels, y_pred))
precision = precision_score(ylabels, y_pred, average='weighted')
recall = recall_score(ylabels, y_pred, average='weighted')
f1 = f1_score(ylabels, y_pred, average='weighted')
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
# make binary executable file
sgd_classifier.fit(X_tfidf, ylabels)
joblib.dump(sgd_classifier, 'code/LOGISTIC_REGRESSION.joblib')
clf = lgb.LGBMClassifier(class_weight='balanced', n_estimators=200, n_jobs=-1)
y_pred = cross_val_predict(clf, X_tfidf, ylabels, cv=10)
cm = confusion_matrix(ylabels, y_pred, labels=np.unique(ylabels))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(ylabels))
disp.plot(cmap='Blues')
print("\nClassification Report:")
print(classification_report(ylabels, y_pred))
precision = precision_score(ylabels, y_pred, average='weighted')
recall = recall_score(ylabels, y_pred, average='weighted')
f1 = f1_score(ylabels, y_pred, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
clf.fit(X_tfidf, ylabels)
joblib.dump(clf, 'code/LIGHTGBM.joblib')

