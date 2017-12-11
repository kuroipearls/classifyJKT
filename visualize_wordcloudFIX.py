from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random
import pandas as pd
import datetime
from dateutil import parser
import re
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
words_array = df1.text
str1 = ' '.join(str(e) for e in words_array)
str1 = str1.replace('bapak', '')
str1 = str1.replace('abang', '')
str1 = str1.replace('cuma', '')
str1 = str1.replace('jika', '')
str1 = str1.replace('bagaimana', '')
str1 = str1.replace('tetapi', '')
str1 = str1.replace('orang', '')
str1 = str1.replace('saat', '')
str1 = str1.replace('jangan', '')
str1 = str1.replace('om', '')
str1 = str1.replace('mpok', '')
str1 = str1.replace('iya', '')
str1 = str1.replace('nanti', '')
str1 = str1.replace('dulu', '')
str1 = str1.replace('sekarang', '')
str1 = str1.replace('mereka', '')
str1 = str1.replace('hanya', '')
str1 = str1.replace('sampai', '')
str1 = str1.replace('kenapa', '')
str1 = str1.replace('memang', '')
str1 = str1.replace('ibu', '')
str1 = str1.replace('siapa', '')
str1 = str1.replace('lo', '')
str1 = str1.replace('ong', '')
str1 = str1.replace('oleh', '')
str1 = str1.replace('terus', '')
str1 = str1.replace('bro', '')
str1 = str1.replace('ayo', '')
str1 = str1.replace('mau', '')
str1 = str1.replace('ingin', '')
str1 = str1.replace('spt', '')
str1 = str1.replace('mah', '')
str1 = str1.replace('seperti', '')
str1 = str1.replace('pasn', '')
str1 = str1.replace('gitu', '')
str1 = str1.replace('pasti', '')
str1 = str1.replace('semua', '')
str1 = str1.replace('kali', '')
str1 = str1.replace('lihat', '')
str1 = str1.replace('mu', '')
str1 = str1.replace('pake', '')
str1 = str1.replace('lain', '')
str1 = str1.replace('agar', '')
str1 = str1.replace('masa', '')
str1 = str1.replace('kl', '')
str1 = str1.replace('cob', '')
str1 = str1.replace('dpt', '')
str1 = str1.replace('begini', '')
str1 = str1.replace('usah', '')
str1 = str1.replace('tanya', '')
str1 = str1.replace('hari', '')
str1 = str1.replace('bs', '')
str1 = str1.replace('sm', '')
str1 = str1.replace('tinggal', '')
str1 = str1.replace('lebih', '')
str1 = str1.replace('dah', '')
str1 = str1.replace('tu', '')
str1 = str1.replace('ngkin', '')
str1 = str1.replace('banget', '')
str1 = str1.replace('demi', '')
str1 = str1.replace('pernah', '')
str1 = str1.replace('hr', '')
str1 = str1.replace('mbak', '')
str1 = str1.replace('nama', '')
str1 = str1.replace('harap', '')
str1 = str1.replace('da', '')
str1 = str1.replace('wah', '')
str1 = str1.replace('mr', '')
str1 = str1.replace('lbh', '')
str1 = str1.replace('bagi', '')
str1 = str1.replace('banyak', '')

str1 = str1.replace('agussylvi', '')

wc = WordCloud(font_path=font_path, background_color="white", max_words=300, max_font_size=300, width=800, height=400)
wc.generate(str1)
wc.recolor(color_func=grey_color_func)

plt.figure(figsize=(12,10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.savefig("agussylviWC4.png", bbox_inches='tight')
plt.show()