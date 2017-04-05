# You need to install scikit-learn:
# sudo pip install scikit-learn
#
# Dataset: Polarity dataset v2.0
# http://www.cs.cornell.edu/people/pabo/movie-review-data/
#
# Full discussion:
# https://marcobonzanini.wordpress.com/2015/01/19/sentiment-analysis-with-python-and-scikit-learn


import sys
import os
import time
import types

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from itertools import chain


classes = [1,-1,0]

# Read the data
train_data = []
train_labels = []
test_data = []
test_labels = []
# for curr_class in classes:
#     dirname = os.path.join(data_dir, curr_class)
#     for fname in os.listdir(dirname):
#         with open(os.path.join(dirname, fname), 'r') as f:
#             content = f.read()
#             if fname.startswith('cv9'):
#                 test_data.append(content)
#                 test_labels.append(curr_class)
#             else:
#                 train_data.append(content)
#                 train_labels.append(curr_class)

with open('dataset/traindata_svm.csv') as data_file:
    for line in data_file:
        train_data.append(line.strip().split(','))
# data = [line.strip() for line in open("dataset/traindata_svm.csv", 'r')]
# train_data = [[word.lower() for word in text.split()] for text in data]

with open('dataset/trainlabels_svm.csv') as data_file:
    for line in data_file:
        train_labels.append(line.strip().split(','))

with open('dataset/testdata_svm.csv') as data_file:
    for line in data_file:
        test_data.append(line.strip().split(','))

with open('dataset/testlabels_svm.csv') as data_file:
    for line in data_file:
        test_labels.append(line.strip().split(','))

train_data = list(chain.from_iterable(train_data))
test_data = list(chain.from_iterable(test_data))
train_labels = list(chain.from_iterable(train_labels))
test_labels = list(chain.from_iterable(test_labels))

# Create feature vectors
#vectorizer = TfidfVectorizer(sublinear_tf=True,use_idf=True,tokenizer=lambda doc: doc, lowercase=False)
#vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
vectorizer = CountVectorizer()
#vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_data)
#print(train_vectors.todense())
#print(train_vectors)
test_vectors = vectorizer.transform(test_data)

# Perform classification with SVM, kernel=rbf
classifier_rbf = svm.SVC(C=1.0, tol=1e-10, cache_size=600, kernel='rbf', gamma='auto', class_weight='balanced')
t0 = time.time()
classifier_rbf.fit(train_vectors, train_labels)
t1 = time.time()
prediction_rbf = classifier_rbf.predict(test_vectors)
t2 = time.time()
time_rbf_train = t1-t0
time_rbf_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(test_vectors)
t2 = time.time()
time_liblinear_train = t1-t0
time_liblinear_predict = t2-t1

#test_labels = [tuple(x) for x in test_labels]
# y = str(''.join(test_labels))# converting list into string
# z = int(y)
# test_labels = z.split()
my_list = [-1, 1, 0]
# test_labels = [list(map(int, row)) for row in test_labels]
# test_labels = [tuple(x) for x in test_labels]
print(test_labels)
print(prediction_rbf)
print(prediction_linear)
# Print results in a nice table
print("Results for SVC(kernel=rbf)")
print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
print(classification_report(test_labels, prediction_rbf))
print(accuracy_score(test_labels, prediction_rbf))
# print("Results for SVC(kernel=linear)")
# print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
# print(classification_report(test_labels, prediction_linear))