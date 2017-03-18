import pandas as pd

test = "aku benci"
test = test.split(' ')
df = pd.read_csv('dataset/testmnb.csv', sep=',')
print(df.values)   # print value of df
#print(df.iloc[0][0])   #test print of a value
#print(df.columns.values)
count_class = len(pd.unique(df[['class']].values.ravel()))     # count how many unique value in column(s) -> to count how many class 
classes = pd.unique(df[['class']].values.ravel())
each_class = pd.value_counts(df[['class']].values.ravel())     # count each value of column(s) 

for clas in classes:
    for word in test: 
        print(df.iloc[0][word])