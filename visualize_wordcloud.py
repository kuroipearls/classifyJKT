from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
import random
from palettable.colorbrewer.sequential import Greys_9

font_path = "font/Mohave.otf"
def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(10, 50)

def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return tuple(Greys_9.colors[random.randint(2,8)])

# csv_path = "label_visualisasi/ahokdjarotfixedit.csv"
df = pd.read_csv('label_visualisasi2/agussylvifixeditSS.csv', sep=',')

words_array = df.text
# words = words_array.to_string()
str1 = ' '.join(str(e) for e in words_array)
# print(str1)
# with open("Output.txt", "w") as text_file:
#     text_file.write(words)

wc = WordCloud(font_path=font_path, background_color="white", max_words=2000, max_font_size=300, width=800, height=400)
wc.generate(str1)
wc.recolor(color_func=color_func)

plt.figure(figsize=(12,10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()