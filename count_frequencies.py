import re
import pandas as pd
from operator import itemgetter

# words = "do do do do do do do do do do re re re re re mi mi fa-so fa fa fa fa fa fa fa fa fa-so fa-so fa-so fa-so fa-so so la ti do"
df = pd.read_csv('label_visualisasi/ahokdjarotfixeditS.csv', sep=',')

words_array = df.text
# words = words_array.to_string()
str1 = ' '.join(str(e) for e in words_array)
print(str1)
item1 = itemgetter(1)

def wordfreq(text):
    d = {}
    for word in re.findall(r"\S+", text):
#    for word in re.findall(r"\w[\w']*", text):
        if word.isdigit():
            continue

        word_lower = word.lower()

        # Look in lowercase dict.
        if word_lower in d:
            d2 = d[word_lower]
        else:
            d2 = {}
            d[word_lower] = d2

        # Look in any case dict.
        d2[word] = d2.get(word, 0) + 1

    d3 = {}
    for d2 in d.values():
        # Get the most popular case.
        first = max(d2.items(), key=item1)[0]
        d3[first] = sum(d2.values())

    return d3.items()

freqs = wordfreq(str1)

print(freqs)