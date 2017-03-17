import csv 

bag_of_words = []
with open('tdm.csv') as csvfile:
    for line in csvfile:
        bag_of_words.append(line.strip().split(','))

print(bag_of_words[1][1])