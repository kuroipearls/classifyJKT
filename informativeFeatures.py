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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

df = pd.read_csv('datasetfixv5/trainingsetfix.csv', sep=',')

X_train = df.text
y_train = df.is_kelas

vect = TfidfVectorizer(binary=True)
X_train_dtm = vect.fit_transform(X_train.values.astype('U'))
# X_train_dtm_array = vect.fit_transform(X_train.values.astype('U')).toarray()
# print(X_train_dtm_array.shape)
clf = LinearSVC()

clf.fit(X_train_dtm, y_train)

def print_top10(vect, clf, class_labels=clf.classes_):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vect.get_feature_names()
    
    for i, class_label in enumerate(class_labels):
        coef = clf.coef_[i].ravel()
        top25 = np.argsort(coef)[-5:]
        y_pos = np.arange(len(top25))
        feature_names = np.array(feature_names)
        plt.figure()
        # print(feature_names[top25])
        if(i == 0):
            colors = 'red'
        elif(i == 1):
            colors = 'green'
        else: 
            colors = 'blue'
        plt.bar(y_pos, coef[top25], align='center', alpha=0.5, color=colors)
        plt.xticks(y_pos, feature_names[top25], rotation=60, ha='right')
        plt.ylabel('Weight')
        if(i == 0):
            # plt.title('Top 5 Features in Class 0')
            plt.savefig('classNeg.png', bbox_inches='tight')
        elif(i == 1):
            # plt.title('Top 5 Features in Class 1')
            plt.savefig('classNet.png', bbox_inches='tight')
        else: 
            # plt.title('Top 5 Features in Class 2')
            plt.savefig('classPos.png', bbox_inches='tight')
        # print(clf.coef_)
        # print("%s: %s" % (class_label,
        #       " ".join(feature_names[j] for j in top10)))

print_top10(vect, clf)