import pandas as pd

df = pd.read_csv('dataset/coba_train.csv', sep=',')
df2 = pd.read_csv('dataset/cobatest.csv', sep=',')
count_train = len(df.index)
print("Training set instances : ", count_train)
count_test = len(df2.index)
print("Testing set instances : ", count_test)
print("Columns : ", len(df.columns))
#print(df2.loc[df2['kelas'] == -1])
#print(df.values)   # print value of df
#print(df.iloc[0][0])   #test print of a value
#print(df.columns.values)
count_class = len(pd.unique(df[['is_kelas']].values.ravel()))     # count how many unique value in column(s) -> to count how many class 
print(count_class)
classes = pd.unique(df[['is_kelas']].values.ravel())              # list of classes -> 0, 1, -1 
print(classes)
each_class = pd.value_counts(df[['is_kelas']].values.ravel())     # count value of each class (columns)
temp_class = 0
temp = 0
count_correct = 0
class_1 = 0
class_2 = 0
class_3 = 0
actual = 0
predicted = 0
conMat = [[0 for x in range(3)] for y in range(3)]
#count MNB (Multinomial Naive Bayes) --> if you want to run the MNB, uncomment line below this. 

for ins in range(0,count_test):
    print("ins : ", ins)
    test = df2.iloc[ins,0]
    test = test.split()
    for num_class in classes:
        #print("kelas : ", num_class)
        result = 1
        cekk = df[(df.is_kelas == num_class)].sum()    # count words in each class -> return series
        cekk = cekk.drop(['is_kelas'])
        words_class = cekk.sum()                    # sum all series
        for word in test:
            #print(df.iloc[0][word])
            if word in list(df.columns.values):
                result = result * (df[(df.is_kelas == num_class)].sum()[word] + 1) / (words_class + (len(df.columns) - 1)) # count per word
            else:
                result = result + 0 # count per word
        result = each_class[num_class] / len(df.index) * result
        #print(result)
        if result > temp:
            temp = result
            temp_class = num_class
    if df2.iloc[ins,1] == 1:
        actual = 0
    elif df2.iloc[ins,1] == -1:
        actual = 1
    else:
        actual = 2
    if temp_class == 1:
        predicted = 0
    elif temp_class == -1:
        predicted = 1
    else:
        predicted = 2

    if temp_class == df2.iloc[ins,1]:
        count_correct += 1
        if temp_class == 1:
            class_1 += 1
            #print("Class 1 : ", df2.iloc[ins,0])
        elif temp_class == -1:
            class_2 += 1
            #print("Class -1 : ", df2.iloc[ins,0])
        else:
            class_3 += 1
    #else:
        #if df2.iloc[ins,i] == 1:
    conMat[actual][predicted] += 1
        #print(df2.iloc[ins,0], " is : ", temp_class, ", should be : ", df2.iloc[ins,1])
accuracy = count_correct / len(df2.index) * 100
print("Accuracy : ", accuracy, " %. ")
for i in range(0,3):
    print("")
    for j in range(0,3):
        print(conMat[i][j], end = " ")
print("")
#print("Test belongs to class : ", temp_class)