import pandas as pd

test = "aku benci"
test = test.split(' ')
df = pd.read_csv('dataset/testmnb.csv', sep=',')
print(df.values)   # print value of df
#print(df.iloc[0][0])   #test print of a value
#print(df.columns.values)
count_class = len(pd.unique(df[['kelas']].values.ravel()))     # count how many unique value in column(s) -> to count how many class 
classes = pd.unique(df[['kelas']].values.ravel())              # list of classes -> 0, 1, -1 
each_class = pd.value_counts(df[['kelas']].values.ravel())     # count value of each class (columns)
temp_class = 0
temp = 0

#count MNB (Multinomial Naive Bayes)
for num_class in classes:
    result = 1
    cekk = df[(df.kelas == num_class)].sum()    # count words in each class -> return series
    cekk = cekk.drop(['kelas'])
    words_class = cekk.sum()                    # sum all series
    for word in test:
        #print(df.iloc[0][word])
        result = result * (df[(df.kelas == num_class)].sum()[word] + 1) / (words_class + (len(df.columns) - 1)) # count per word
    result = each_class[num_class] / len(df.index) * result
    print(result)
    if result > temp:
        temp = result
        temp_class = num_class
print("Test belongs to class : ", temp_class)