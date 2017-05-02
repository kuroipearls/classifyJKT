import pandas as pd
import numpy as np
import time
from pandas import DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC

df = pd.read_csv('dataset/cobacektrain.csv', sep=',')
df2 = pd.read_csv('dataset/cobatest.csv', sep=',')

X_train = df.text
y_train = df.is_kelas
X_test = df2.text
y_test = df2.is_kelas

vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
# print(X_train_dtm)
X_test_dtm = vect.transform(X_test)
# print(X_test_dtm)

##### USING MNB #####

nb = MultinomialNB()
# rfe = RFECV(estimator = nb, cv = 20, scoring="accuracy")
# rfe = rfe.fit(X_train_dtm, y_train)
# print("Optimal number of features : %d" % rfe.n_features_)

##### END USING MNB ######

##### USING KNN #####

# knn = KNeighborsClassifier(n_neighbors=103)
# rfe = RFE(knn, 2430)
# rfe = rfe.fit(X_train_dtm, y_train)

##### END USING KNN #####

##### USING SVM #####

svr = LinearSVC()
# svr = svm.SVC(kernel="rbf",C=3.0,gamma=0.05)
rfe = RFECV(estimator = svr, cv = 5, scoring="accuracy")
rfe = rfe.fit(X_train_dtm, y_train)
print("Optimal number of features : %d" % rfe.n_features_)

##### END USING SVM #####

# print(rfe.support_)
# print(rfe.ranking_)


y_pred_class = rfe.predict(X_test_dtm)
print(metrics.accuracy_score(y_test, y_pred_class))
print(y_test.value_counts())
print(metrics.confusion_matrix(y_test, y_pred_class))