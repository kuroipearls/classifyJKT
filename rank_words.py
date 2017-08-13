from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

df = pd.read_csv('datasetfix/trainingsetfix.csv', sep=',')
df2 = pd.read_csv('datasetfix/testingsetfix.csv', sep=',')

X_train = df.text
y_train = df.is_kelas
X_test = df2.text
y_test = df2.is_kelas
# print(X_train)

vect = TfidfVectorizer(min_df=1)
X_train_dtm = vect.fit_transform(X_train.values.astype('U'))
# print(X_train_dtm)
X_test_dtm = vect.transform(X_test.values.astype('U'))

feature_names = vect.get_feature_names()
print(len(feature_names))
indices = np.argsort(vect.idf_)[::1]
top_n = 100

top_features = [feature_names[i] for i in indices[:top_n]]
# print(top_features)