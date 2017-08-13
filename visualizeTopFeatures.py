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
import matplotlib.pyplot as plt
start_time = time.time()

def plot_coefficients(classifier, feature_names, top_features=20):
 coef = classifier.coef_.ravel()
 top_positive_coefficients = np.argsort(coef)[-top_features:]
 top_negative_coefficients = np.argsort(coef)[:top_features]
 top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
 # create plot
 plt.figure(figsize=(15, 5))
 colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
 plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
 feature_names = np.array(feature_names)
 plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
 plt.show()

df = pd.read_csv('dataset/trainNormalAniesDes3.csv', sep=',')
df2 = pd.read_csv('dataset/testNormalAniesDes3.csv', sep=',')

X_train = df.text
y_train = df.is_kelas
X_test = df2.text
y_test = df2.is_kelas

vect = TfidfVectorizer(min_df=1)
# vect = CountVectorizer()
# print(len(vect.vocabulary_))
X_train_dtm = vect.fit_transform(X_train)
# print(X_train_dtm)
X_test_dtm = vect.transform(X_test)
# print(X_test_dtm)

svr = LinearSVC()
# svr = svm.SVC(kernel="rbf",C=3.0,gamma=0.05)
# rfe = RFECV(estimator = svr, cv = 10, scoring="accuracy")
# rfe = rfe.fit(X_train_dtm, y_train)
svr.fit(X_train_dtm, y_train)
# print("Optimal number of features : %d" % rfe.n_features_)
# y_pred_class = rfe.predict(X_test_dtm)

# print(metrics.accuracy_score(y_test, y_pred_class))
# print(y_test.value_counts())
# print(metrics.confusion_matrix(y_test, y_pred_class))
print("--- %s seconds ---" % (time.time() - start_time))
plot_coefficients(svr, vect.get_feature_names())