# import numpy as np
# import pandas as pd
# X = np.random.randint(5, size=(6, 100))
# y = np.array([1,2,3,4,5,6])
# df2 = pd.read_csv('dataset/cobatest.csv', sep=',')
# print("X = ", X)
# print("Y = ", y)
# print(df2)
# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB()
# clf = clf.fit(X,y)
# print(clf.predict(X))

##### FIRST TRY (SPLIT IN CODE) #####

# import pandas as pd
# import numpy as np
# from pandas import DataFrame
# from sklearn.cross_validation import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn import metrics
# df2 = pd.read_csv('dataset/cobatest.csv', sep=',')
# # print(df2.shape)
# # print(df2.head)
# X = df2.text
# y = df2.is_kelas
# # print(X.shape)
# # print(y.shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# vect = CountVectorizer()
# X_train_dtm = vect.fit_transform(X_train)
# # print(X_train_dtm)
# X_test_dtm = vect.transform(X_test)
# # print(X_test_dtm)
# # print(DataFrame(X_train_dtm.A, columns=vect.get_feature_names()).to_string())

# nb = MultinomialNB()
# nb.fit(X_train_dtm, y_train)
# y_pred_class = nb.predict(X_test_dtm)
# print(metrics.accuracy_score(y_test, y_pred_class))

##### END FIRST TRY #####

##### SECOND TRY (SPLIT MANUALLY) #####
import pandas as pd
import numpy as np
import time
from pandas import DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
start_time = time.time()
df = pd.read_csv('dataset/cobacektrain.csv', sep=',')
df2 = pd.read_csv('dataset/cobatest.csv', sep=',')
# df['is_kelass'] = df.is_kelas.map({0:0, 1:1, -1:2})
# df2['is_kelass'] = df2.is_kelas.map({0:0, 1:1, -1:2})
# print(df2.shape)
# print(df2.head)
X_train = df.text
y_train = df.is_kelas
X_test = df2.text
y_test = df2.is_kelas
# print(X.shape)
# print(y.shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
# print(X_train_dtm)
X_test_dtm = vect.transform(X_test)
# print(X_test_dtm)
# print(DataFrame(X_train_dtm.A, columns=vect.get_feature_names()).to_string())

### USING MNB ###
# nb = MultinomialNB()
# nb.fit(X_train_dtm, y_train)
# y_pred_class = nb.predict(X_test_dtm)
# print(metrics.accuracy_score(y_test, y_pred_class))
# print(y_test.value_counts())
# print(metrics.confusion_matrix(y_test, y_pred_class))

### END USING MNB ###

### USING KNN ###
# neigh = KNeighborsClassifier(n_neighbors=103)
# neigh.fit(X_train_dtm, y_train)
# y_pred_class = neigh.predict(X_test_dtm)
# print(metrics.accuracy_score(y_test, y_pred_class))
# print(y_test.value_counts())
# print(metrics.confusion_matrix(y_test, y_pred_class))
### END USING KNN ###

### USING SVM (WITH TUNING) ###
parameters = {'kernel':('linear', 'rbf'), 'C':[1,2,3,4,5,6,7,8,9,10], 'gamma': 
[0.01,0.02,0.03,0.04,0.05,0.10,0.2,0.3,0.4,0.5]}
svr = svm.SVC()
grid = GridSearchCV(svr, parameters)
grid.fit(X_train_dtm,y_train)
y_pred_class = grid.predict(X_test_dtm)
print('Best C:',grid.best_estimator_.C)
print('Best Kernel:',grid.best_estimator_.kernel)
print('Best Gamma:',grid.best_estimator_.gamma)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred_class))
print(y_test.value_counts())
print(metrics.confusion_matrix(y_test, y_pred_class))
print("--- %s seconds ---" % (time.time() - start_time))
### END USING SVM ###

### USING SVM (WITH TUNING) ###
# parameters = {'C':[1,2,3,4,5,6,7,8,9,10], 'dual': [True,False]}
# svr = LinearSVC()
# grid = GridSearchCV(svr, parameters)
# grid.fit(X_train_dtm,y_train)
# y_pred_class = grid.predict(X_test_dtm)
# print('Best C:',grid.best_estimator_.C)
# print('Best Dual:',grid.best_estimator_.dual)
# print('Accuracy: ',metrics.accuracy_score(y_test, y_pred_class))
# print(y_test.value_counts())
# print(metrics.confusion_matrix(y_test, y_pred_class))
# print("--- %s seconds ---" % (time.time() - start_time))
### END USING SVM ###

##### END SECOND TRY #####