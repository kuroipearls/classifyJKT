import datetime
import pandas as pd
from dateutil import parser

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
from plotly.graph_objs import Scatter, Figure, Layout 
# import plotly
# plotly.offline.init_notebook_mode()
# import plotly.plotly as py
# import plotly.graph_objs as go


# dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M')
# df = pd.read_csv('label_visualisasi/ahokdjarotfixeditS.csv', sep=',', parse_dates=['date'], date_parser=dateparse)
# # df = pd.read_csv('label_visualisasi/agussylvifixedit.csv', sep=',')
# # pd.DatetimeIndex(df.date).normalize()
# # df['date_fix'] = pd.DatetimeIndex(df.date).normalize()
# # df['date_fix']= pd.to_datetime(df['date'])
# df['just_date'] = df['date'].dt.date
# # df['date_fix'] = df['date_fix'].apply(lambda x:x.date().strftime('%d/%m/%y'))
# df['count'] = 1

# daily_counts = df.groupby(by=['is_kelas', 'just_date']).count()
# daily_counts_xtab = daily_counts.unstack(level='is_kelas')['count']
# daily_counts_xtab = daily_counts_xtab.reset_index('date')
# daily_counts_xtab.columns = ['date','negative','neutral','positive']
# print(daily_counts_xtab.head())
# # daily_counts_xtab.to_csv("cek_tweet.csv", sep=',', encoding='utf-8')
# print(type(daily_counts_xtab))

# # df2 = df.groupby(["is_kelas", "just_date"]).size().reset_index(name='count')
# # print(df2)

# # df2 = pd.read_csv('cek_tweet.csv', sep=',')
# # df['date'] = pd.Series([val.date() for val in df['date']])

# # print(isinstance(df.date, str))
# # print(type(df.date))

# # plot([Scatter(x=df2['just_date'],y=df2['positive'])])

# # data = [Scatter(x=df2['just_date'],y=df2['positive'])]

# # iplot(data)

dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M')
df1 = pd.read_csv('label_visualisasi/agussylvifixeditS.csv', sep=',', parse_dates=['date'], date_parser=dateparse)
df2 = pd.read_csv('label_visualisasi/ahokdjarotfixeditS.csv', sep=',', parse_dates=['date'], date_parser=dateparse)
df3 = pd.read_csv('label_visualisasi/aniessandifixeditS.csv', sep=',', parse_dates=['date'], date_parser=dateparse)
df1['just_date'] = df1['date'].dt.date
df2['just_date'] = df2['date'].dt.date
df3['just_date'] = df3['date'].dt.date
df1['count'] = 1
df2['count'] = 1
df3['count'] = 1

dcounts1 = df1.groupby(by=['sentimen_1', 'just_date']).count()
dfcounts1 = dcounts1.unstack(level='sentimen_1')['count']
dfcounts1 = dfcounts1.reset_index('date')
dfcounts1.columns = ['date','negative','neutral','positive']

dcounts2 = df2.groupby(by=['sentimen_2', 'just_date']).count()
dfcounts2 = dcounts2.unstack(level='sentimen_2')['count']
dfcounts2 = dfcounts2.reset_index('date')
print(dfcounts2)
dfcounts2.columns = ['date','negative','neutral','positive']

dcounts3 = df3.groupby(by=['sentimen_3', 'just_date']).count()
dfcounts3 = dcounts3.unstack(level='sentimen_3')['count']
dfcounts3 = dfcounts3.reset_index('date')
dfcounts3.columns = ['date','negative','neutral','positive']