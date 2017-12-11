import pandas as pd
import numpy as np
import time
from pandas import DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
import matplotlib.pyplot as plt
start_time = time.time()
df = pd.read_csv('dataset_iseng/training.csv', sep=',')
print(df.columns)
df2 = pd.read_csv('dataset_iseng/testing.csv', sep=',')

X_train = df.text
y_train = df.is_kelas
X_test = df2.text
y_test = df2.is_kelas

vect = TfidfVectorizer(min_df=1)
X_train_dtm = vect.fit_transform(X_train.values.astype('U'))
# print(X_train_dtm)
X_test_dtm = vect.transform(X_test.values.astype('U'))
# print(X_test_dtm)

##### USING MNB #####

# nb = MultinomialNB()
# rfe = RFECV(estimator = nb, cv = 10, scoring="accuracy")
# rfe = rfe.fit(X_train_dtm, y_train)
# print("Optimal number of features : %d" % rfe.n_features_)

##### END USING MNB ######

##### USING KNN #####

# knn = KNeighborsClassifier(n_neighbors=103)
# rfe = RFE(knn, 2430)
# rfe = rfe.fit(X_train_dtm, y_train)

##### END USING KNN #####

#### USING SVM #####

svr = LinearSVC(C=1.0, multi_class='ovr')
svr = svr.fit(X_train_dtm, y_train)
print(svr.intercept_)
print(svr.decision_function(X_test_dtm))
print(svr.predict(X_test_dtm))

# svr = svm.SVC(kernel="rbf",C=3.0,gamma=0.05)
# rfe = RFECV(estimator = svr, cv = 5, scoring="accuracy")
# rfe = rfe.fit(X_train_dtm, y_train)
# print("Optimal number of features : %d" % rfe.n_features_)

#### END USING SVM #####

# ### USING SVM - VERSION 2 ###
# estimator = LinearSVC(C=3.0, dual=True)
# rfe = RFE(estimator, 9266, step=40)
# rfe = rfe.fit(X_train_dtm, y_train)
# #### END USING SVM  - VERSION 2#####

# print(rfe.support_)
# print(rfe.ranking_)

# jangan lupa di uncomment
# y_pred_class = rfe.predict(X_test_dtm)

# misData = [teks
#           for teks, truth, prediction in
#           zip(X_test, y_test, y_pred_class)
#           if truth != prediction]


# misDataTeks = [teks
#           for teks, truth, prediction in
#           zip(X_test, y_test, y_pred_class)
#           if truth != prediction]
# misDataTruth = [truth
#           for teks, truth, prediction in
#           zip(X_test, y_test, y_pred_class)
#           if truth != prediction]
# misDataPred = [prediction
#           for teks, truth, prediction in
#           zip(X_test, y_test, y_pred_class)
#           if truth != prediction]
# misDFTeks = pd.DataFrame(misDataTeks)
# misDFTruth = pd.DataFrame(misDataTruth)
# misDFPred = pd.DataFrame(misDataPred)


# misDF = pd.DataFrame(misData)
# misDF.to_csv("aniesDesError5.csv")
# misDFTeks.to_csv("aniesDesError5Teks.csv")
# misDFTruth.to_csv("aniesDesError5Truth.csv")
# misDFPred.to_csv("aniesDesError5Pred.csv")

# jangan lupa di uncomment
# print(metrics.accuracy_score(y_test, y_pred_class))
# print(y_test.value_counts())
# print(metrics.confusion_matrix(y_test, y_pred_class))
# print("--- %s seconds ---" % (time.time() - start_time))

# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("CV Score")
# plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
# plt.show()
