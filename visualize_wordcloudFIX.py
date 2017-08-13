from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random
import pandas as pd
import datetime
from dateutil import parser
from palettable.colorbrewer.sequential import Greys_9
from palettable.colorbrewer.sequential import Reds_9
from palettable.colorbrewer.sequential import Blues_9

dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M')
df1 = pd.read_csv('label_visualisasi3/agussylvifixeditSS.csv', sep=',', parse_dates=['date'], date_parser=dateparse)
df2 = pd.read_csv('label_visualisasi3/ahokdjarotfixeditSS.csv', sep=',', parse_dates=['date'], date_parser=dateparse)
df3 = pd.read_csv('label_visualisasi3/aniessandifixeditSS.csv', sep=',', parse_dates=['date'], date_parser=dateparse)
font_path = "font/Mohave.otf"

def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return tuple(Greys_9.colors[random.randint(2,8)])

def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return tuple(Reds_9.colors[random.randint(2,8)])

def blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return tuple(Blues_9.colors[random.randint(2,8)])

#make wordcloud for the Candidate 1 
words_array = df3.text
str1 = ' '.join(str(e) for e in words_array)

wc = WordCloud(font_path=font_path, background_color="white", max_words=2000, max_font_size=300, width=800, height=400)
wc.generate(str1)
wc.recolor(color_func=blue_color_func)

plt.figure(figsize=(12,10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.savefig("aniessandiWC2.png", bbox_inches='tight')
plt.show()