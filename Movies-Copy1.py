#!/usr/bin/env python
# coding: utf-8

# Эти файлы содержат метаданные для всех 45 000 фильмов, перечисленных в полном наборе данных MovieLens. Набор данных состоит из фильмов, выпущенных до или до июля 2017 года. Точки данных включают состав, команду, ключевые слова сюжета, бюджет, доход, плакаты, даты выхода, языки, производственные компании, страны, подсчет голосов TMDB и средние значения голосов.
# 
# Этот набор данных также содержит файлы, содержащие 26 миллионов оценок от 270 000 пользователей для всех 45 000 фильмов. Рейтинги по шкале от 1 до 5 получены на официальном сайте GroupLens.
# 
# Этот набор данных состоит из следующих файлов:
# 
# <b>movies_metadata.csv:</b> основной файл метаданных фильмов. Содержит информацию о 45 000 фильмов, представленных в наборе данных Full MovieLens. Возможности включают постеры, фоны, бюджет, доходы, даты выпуска, языки, страны производства и компании.
# 
# <b>keywords.csv:</b> содержит ключевые слова сюжета для наших фильмов MovieLens. Доступно в виде строкового объекта JSON.
# 
# <b>credits.csv:</b> содержит информацию о ролях и съемочной группе для всех наших фильмов. Доступно в виде строкового объекта JSON.
# 
# <b>links.csv:</b> файл, содержащий идентификаторы TMDB и IMDB для всех фильмов, представленных в наборе данных Full MovieLens.
# 
# <b>links_small.csv:</b> содержит идентификаторы TMDB и IMDB для небольшого подмножества из 9 000 фильмов полного набора данных.
# 
# <b>rating_small.csv:</b> подмножество 100 000 оценок от 700 пользователей на 9 000 фильмов.

# adult: Indicates if the movie is X-Rated or Adult.
# 
# belongs_to_collection: A stringified dictionary that gives information on the movie series the particular film belongs to.
# 
# budget: The budget of the movie in dollars.
# 
# genres: A stringified list of dictionaries that list out all the genres associated with the movie.
# 
# homepage: The Official Homepage of the move.
# 
# id: The ID of the move.
# imdb_id: The IMDB ID of the movie.
# 
# original_language: The language in which the movie was originally shot in.
# 
# original_title: The original title of the movie.
# 
# overview: A brief blurb of the movie.
# 
# popularity: The Popularity Score assigned by TMDB.
# 
# poster_path: The URL of the poster image.
# 
# production_companies: A stringified list of production companies involved with the making of the movie.
# 
# production_countries: A stringified list of countries where the movie was shot/produced in.
# 
# release_date: Theatrical Release Date of the movie.
# 
# revenue: The total revenue of the movie in dollars.
# 
# runtime: The runtime of the movie in minutes.
# 
# spoken_languages: A stringified list of spoken languages in the film.
# 
# status: The status of the movie (Released, To Be Released, Announced, etc.)
# 
# tagline: The tagline of the movie.
# 
# title: The Official Title of the movie.
# 
# video: Indicates if there is a video present of the movie with TMDB.
# 
# vote_average: The average rating of the movie.
# 
# vote_count: The number of votes by users, as counted by TMDB.

# In[1]:


# отключим предупреждения Anaconda
import warnings
warnings.simplefilter('ignore')

# будем отображать графики прямо в jupyter'e
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
#графики в svg выглядят более четкими
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

#увеличим дефолтный размер графиков
from pylab import rcParams
rcParams['figure.figsize'] = 8, 5
import pandas as pd
import  numpy as np


# ## Загрузка данных и первичная обработка

# In[2]:


movies_metadata = pd.read_csv('movies_metadata.csv')
#keywords = pd.read_csv('keywords.csv')
#credits = pd.read_csv('credits.csv')
links_small = pd.read_csv('links_small.csv')
ratings_small = pd.read_csv('ratings_small.csv')


# In[3]:


movies_metadata.info()


# In[4]:


movies = movies_metadata.loc[:, ['adult', 'budget', 'genres', 'id', 'imdb_id', 'original_language', 'popularity', 'production_companies',
                        'production_countries', 'release_date', 'revenue', 'runtime', 'title', 'spoken_languages', 'status']]
movies = movies.dropna()


# In[5]:


movies['budget'] = movies['budget'].astype('float')
movies['adult'] = movies['adult'].astype('bool')
movies['genres'] = movies['genres'].astype('str')
movies['original_language'] = movies['original_language'].astype('str')
movies['popularity'] = movies['popularity'].astype('float64')
movies['production_companies'] = movies['production_companies'].astype('str')
movies['production_countries'] = movies['production_countries'].astype('str')
movies['release_date'] = movies['release_date'].astype('str')
movies['revenue'] = movies['revenue'].astype('float')
movies['runtime'] = movies['runtime'].astype('float')
movies['title'] = movies['title'].astype('str')
movies['spoken_languages'] = movies['spoken_languages'].astype('str')
movies['status'] = movies['status'].astype('str')


# In[6]:


movies['imdbId'] = 0
movies['genres_per_one'] = 0
movies['count_of_genres'] = 0
movies['year'] = 0
ll = []
for i in range(movies.shape[0]):
    #movies.loc[i, 'imdbId'] = int(str(movies.iloc[i]['imdb_id'])[3:])
    l = []
    s = ''
    for j in range(1, len(movies.iloc[i]['genres'][1:-1].split(',')), 2):
        l.append(movies.iloc[i]['genres'][1:-1].split(',')[j].split("'")[-2])
    for e in sorted(l):
        s += e + '_'
        ll.append(e)
    movies.loc[i, 'genres_per_one'] = s[:-1]
    movies.loc[i, 'count_of_genres'] = len(str(s[:-1]).split('_'))
    movies.loc[i, 'year'] = int(str(movies.iloc[i].release_date).split('-')[0])
all_genres = set(ll)


# In[7]:


for name in all_genres:
    movies[name] = False
for i in range(movies.shape[0]):
    for e in str(movies.iloc[i]['genres_per_one']).split('_'):
        movies.loc[i, e] = True


# In[8]:


movies = movies.drop('genres', axis=1)


# In[9]:


count_genre = []
all_genr = list(all_genres)
for name in all_genr:
    count_genre.append(movies[movies[name]==True].shape[0])


# ## Топы по всем данным 

# In[10]:


rcParams['figure.figsize'] = 8, 6
plt.barh(all_genr, count_genre, color='pink')
plt.title('Количество фильмов по жанрам')


# In[11]:


con = (movies.status=='Released') & (movies.budget != 0)
movies[con].groupby(by='original_language').count().sort_values(
    ['adult'], ascending=False).adult[:20].plot(kind='bar', color='pink')


# In[12]:


con = (movies.status=='Released') & (movies.budget != 0)
movies[con].groupby(by='original_language').count().sort_values(
    ['adult'], ascending=False).adult[1:20].plot(kind='bar', color='pink')


# In[13]:


con = (movies.status=='Released') & (movies.budget != 0)
movies[con].groupby(by='original_language')[['revenue']].sum().sort_values(
    ['revenue'], ascending=False).revenue[1:20].plot(kind='bar', color='pink')
plt.title('Топ языков фильмов с самыми большими сборами в совокупности')


# In[14]:


plt.barh(movies.sort_values('revenue', ascending=False).head(20).title,
         movies.sort_values('revenue', ascending=False).head(20).revenue, color='pink')
plt.title('Топ фильмов с самыми большими сборами')


# In[15]:


plt.barh(movies.sort_values('budget', ascending=False).head(20).title,
         movies.sort_values('budget', ascending=False).head(20).budget, color='pink')
plt.title('Топ фильмов с самым большим бюджетом')


# In[16]:


plt.barh(movies.sort_values('popularity', ascending=False).head(20).title,
         movies.sort_values('popularity', ascending=False).head(20).popularity, color='pink')
plt.title('Топ самых популярных фильмов')


# ## Распределения всякие там 

# In[17]:


sns.distplot(movies[~np.isnan(movies.popularity)].popularity, color='pink')
plt.title('Распределение оценок популярности')


# In[200]:


min(sorted(movies.popularity.unique()[:-1])), max(sorted(movies.popularity.unique()[:-1]))


# In[18]:


con = (movies.revenue == movies.revenue) & (movies.revenue != 0)
sns.distplot(movies[con].revenue, color='pink')


# In[141]:


con = ~np.isnan(movies.revenue) & (movies.revenue > movies.revenue.quantile(0.05)) & (movies.revenue < movies.revenue.quantile(0.95)) & (movies.revenue > 0)
#con = (movies.revenue == movies.revenue) & (movies.revenue != 0)
sns.distplot(movies[con].revenue, color='pink')


# In[57]:


sns.distplot(movies[~np.isnan(movies.runtime)].runtime, color='pink')
plt.title('Распределение по длительности фильмов')


# In[59]:


con = ~np.isnan(movies.runtime) & (movies.runtime > movies.runtime.quantile(0.05)) & (movies.runtime < movies.runtime.quantile(0.95))
sns.distplot(movies[con].runtime, color='pink')
plt.title('Распределение по длительности фильмов')


# In[174]:


word_mov = movies[movies.title == movies.title]
d = {}
for tit in list(word_mov.title):
    for e in tit.split():
        if e in d.keys():
            d[e]+=1
        else:
            d[e]=1


# In[214]:


rev_d = {}
for key, val in d.items():
    rev_d[val] = key
sort_d = {}
for e in sorted(rev_d.keys())[::-1]:
    sort_d[e] = rev_d[e]
count_of_words=[]
words = []
new_words = {}
for key, val in sort_d.items():
    if val.lower()=='the' or val.lower()=='and' or val.lower()=='in' or val.lower()=='to' or val.lower()=='a' or val.lower()=='an' or val.lower()=='of' or val=='for' or val=='&' or val=='on' or str(val)=='2' or val=='In' or val=='from' or val=='Is' or val=='at' or val=='with':
        continue
    count_of_words.append(key)
    words.append(val)
    new_words[val] = key


# In[215]:


plt.bar(words[:17], count_of_words[:17], color='pink')
plt.title('Самые частые содержательные слова в названии')


# In[219]:


#from wordcloud import WordCloud
#tone = 100 # define the color of the words
#f, ax = plt.subplots(figsize=(14, 6))
#wordcloud = WordCloud(width=550,height=300, background_color='black', 
#                      max_words=1628, relative_scaling=0.7,
#                      color_func = random_color_func,
#                      normalize_plurals=False)
#wordcloud.generate_from_frequencies(new_words)
#plt.imshow(wordcloud, interpolation="bilinear")
#plt.axis('off')
#plt.show()


# ## Раcпределение по ДФ с рейтингами 

# In[19]:


sns.distplot(ratings_small[~np.isnan(ratings_small.rating)].rating, color='pink')
plt.title('Распределение оценок зрителей(рейтинг)')


# ## Добавление инфы о среднем рейтинге в ДФ с фильмами 

# In[22]:


# сделали таблицу с рейтингами и всеми ключами по фильмам, чтоб легче было склеивать потом с таблицей основной.
rating = pd.merge(ratings_small, links_small, on='movieId')
rating['imdbId'] = rating['imdbId'].astype('int')


# In[23]:


movies['rating_mean']=0
for i in range(movies[movies.imdb_id == movies.imdb_id].shape[0]):
    movies.loc[i, 'rating_mean'] = rating[rating.imdbId==int(movies.iloc[i].imdb_id[3:])]['rating'].mean()


# In[24]:


con = (movies.rating_mean != 0) & (~np.isnan(movies.rating_mean))
sns.distplot(movies[con].rating_mean, color='pink')
plt.title('Распределение оценок фильмов из ДФ')


# In[26]:


print('Несколько самых высоко-оцененные фильмов')
movies.sort_values('rating_mean', ascending=False).head(20).title


# In[27]:


movies = movies.drop('0', axis=1)
movies = movies.drop('', axis=1)
movies_clean = movies.dropna()
#movies_clean.info()


# In[28]:


movies['revenue_budget_ratio'] = movies['revenue']/movies['budget']


# In[31]:


print('Распределение по среднему рейтингу и коэффициенту доходности')
sns.jointplot(x='revenue_budget_ratio', y='rating_mean', data=movies[movies.budget>0], color='pink')


# ## Исследование в разрезе временных лет 

# In[32]:


con = (~np.isnan(movies.year)) & (movies.year != 0)
mov = movies.copy()
year_mov = mov[con]
year_count = year_mov.groupby('year')['title'].count()
plt.figure(figsize=(8,4))
year_count.plot(rot=45, color='pink')
plt.title('Количество фильмов по годам')


# In[33]:


year_mov['kagorta'] = 0
for i in range(year_mov.shape[0]):
    if int(year_mov.iloc[i].year) <= 1920:
        year_mov.loc[i, 'kagorta'] = 'untill 1920'
    elif int(year_mov.iloc[i].year) > 1920 and int(year_mov.iloc[i].year) <= 1940:
        year_mov.loc[i, 'kagorta'] = '1920-1940'
    elif int(year_mov.iloc[i].year) > 1940 and int(year_mov.iloc[i].year) <= 1960:
        year_mov.loc[i, 'kagorta'] = '1940-1960'
    elif int(year_mov.iloc[i].year) > 1960 and int(year_mov.iloc[i].year) <= 1980:
        year_mov.loc[i, 'kagorta'] = '1960-1980'
    elif int(year_mov.iloc[i].year) > 1980 and int(year_mov.iloc[i].year) <= 2000:
        year_mov.loc[i, 'kagorta'] = '1980-2000'
    
    elif int(year_mov.iloc[i].year) > 2000 and int(year_mov.iloc[i].year) <= 2020:
        year_mov.loc[i, 'kagorta'] = '2000-2020'


# In[34]:


li = ['budget', 'revenue']
kagorta_df = year_mov[[x for x in year_mov.columns if x in li] + ['kagorta']]
kagorta_df.groupby('kagorta').sum().plot(kind='bar', rot=45, colormap='Pastel1')
plt.title('Суммарные бюджет и сборы по кагортам')


# In[35]:


kagorta_df.groupby('kagorta')[['budget']].count().plot.pie(y='budget', figsize=(5, 8), colormap='Set3')
plt.title('Число фильмов в разрезе кагорт')


# In[46]:


year_mov.loc[:, all_genr + ['kagorta']].groupby(by='kagorta').sum().plot(kind='bar', rot=45)
plt.title('Количество фильмов в разрезе жанров и кагорт')


# In[47]:


ration = year_mov.drop('rating_mean', axis=1).dropna()


# In[48]:


ration.loc[:, ['popularity', 'kagorta']].groupby(by='kagorta').sum().plot(kind='bar', rot=45, color='pink')
plt.title('Популярность по кагортам')


# ## "На 2 стульях не усидишь" или выборка одножанровых фильмов 

# In[49]:


movie_one_genre = year_mov.copy()
movie_one_genre = movie_one_genre[movie_one_genre.count_of_genres==1]


# In[50]:


for name in all_genr:
    try:
        movie_one_genre = movie_one_genre.drop(name, axis=1)
    except:
        pass
movie_one_genre.columns


# In[51]:


movie_one_genre = movie_one_genre.dropna()
movie_one_genre.info()


# In[52]:


movie_one_genre['revene_budget_ratio'] = movie_one_genre['revenue']/movie_one_genre['budget']


# In[53]:


movie_one_genre.groupby(by=['genres_per_one']).count().id.plot(kind='bar', color='pink')
plt.title('Распределение по жанрам среди одножанровых')


# In[54]:


movie_one_genre_2 = year_mov[year_mov.count_of_genres==1]
movie_one_genre.info()


# In[82]:


con = (movie_one_genre_2.year>=2000) & (movie_one_genre_2.year!=2018)
movie_one_genre_2.loc[con, all_genr+[ 'year']].groupby(by='year').sum().plot()
plt.title('Число фильмов в разреpе жанров в 2000-2017')


# In[84]:


con = (movie_one_genre_2.year>=1980) & (movie_one_genre_2.year < 2000)
movie_one_genre_2.loc[con, all_genr+[ 'year']].groupby(by='year').sum().plot()
plt.title('Число фильмов в разреpе жанров в 1980-2000')


# In[86]:


con = (movie_one_genre_2.year >= 1940) & (movie_one_genre_2.year < 1960)
movie_one_genre_2.loc[con, all_genr+[ 'year']].groupby(by='year').sum().plot()
plt.title('Число фильмов в разреpе жанров до 1940-1960')


# In[117]:


con = (movie_one_genre_2.year>=2000) & (movie_one_genre_2.year!=2018)
dat = movie_one_genre_2.loc[con, all_genr+[ 'year']].groupby(by='year').sum()
dat['count'] = 0
for i in dat.index:
    dat.loc[i, 'count'] = 0
    for name in dat.columns:
        dat.loc[i, 'count'] += int(dat.loc[i, name])
#plt.title('Число фильмов в разреpе жанров в 2000-2017')


# In[118]:


for i in dat.index:
    for name in dat.columns:
        dat.loc[i, name] = int(dat.loc[i, name])/dat.loc[i, 'count']*100


# In[120]:


new_dat = dat.copy()
new_dat = new_dat.drop('count', axis=1)
new_dat.plot()
plt.title('Процент в разреpе жанров в 2000-2017')


# In[127]:


rcParams['figure.figsize'] = 10, 8
ax = sns.heatmap(data=new_dat, annot=True, fmt=".1f", linewidths=.5, cmap="YlGnBu")


# In[155]:


ym = year_mov.copy()
ym = ym.dropna()
l = ['budget', 'original_language', 'popularity','year', 'count_of_genres', 'genres_per_one']
ym = ym.loc[:, l]
ym.info()


# In[160]:


con = (ym.genres_per_one != '') & (ym.count_of_genres==1)
ym['genres_per_one'] = ym['genres_per_one'].astype('str')
ym[con].info()


# In[162]:


yymm = ym.loc[con,  ['budget', 'original_language', 'popularity','year', 'genres_per_one']]
sns.pairplot(yymm, hue='genres_per_one')


# In[165]:


con = (ym.genres_per_one != '') & (ym.count_of_genres==1) & ((ym.genres_per_one =='Drama') | (ym.genres_per_one =='Comedy') | (ym.genres_per_one == 'Documentary') | (ym.genres_per_one == 'Horror'))
yymm = ym.loc[con,  ['budget', 'original_language', 'popularity','year', 'genres_per_one']]
sns.pairplot(yymm, hue='genres_per_one')


# In[227]:


my = movies.copy()
my = movies.dropna()


# In[241]:


for g in list(my[my.count_of_genres==1].genres_per_one.unique()):
    if g== '':
        continue
    count_of_words, words, new_words = get_pop_word(my[my[g] == True])
    print(f'{g}: {new_words}')


# In[222]:


def get_pop_word(movies):
    word_mov = movies[movies.title == movies.title]
    d = {}
    for tit in list(word_mov.title):
        for e in tit.split():
            if e in d.keys():
                d[e]+=1
            else:
                d[e]=1
    rev_d = {}
    for key, val in d.items():
        rev_d[val] = key
    sort_d = {}
    for e in sorted(rev_d.keys())[::-1]:
        sort_d[e] = rev_d[e]
    count_of_words=[]
    words = []
    new_words = {}
    for key, val in sort_d.items():
        if val.lower()=='the' or val.lower()=='and' or val.lower()=='in' or val.lower()=='to' or val.lower()=='a' or val.lower()=='an' or val.lower()=='of' or val=='for' or val=='&' or val=='on' or str(val)=='2' or val=='In' or val=='from' or val=='Is' or val=='at' or val=='with':
            continue
        count_of_words.append(key)
        words.append(val)
        new_words[val] = key
    return count_of_words, words, new_words


# In[ ]:





# ## Рисунки на полях 

# In[320]:


movie_one_genre['imdbId'] = 0
for i in range(movie_one_genre.shape[0]):
    movie_one_genre.loc[i, 'imdbId'] = int(str(movie_one_genre.iloc[i]['imdb_id'])[3:])


# In[57]:


one_genre = movie_rel[movie_rel.count_of_genres == 1]
one_genre['budget'] = one_genre['budget'].astype('int')


# In[66]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go

init_notebook_mode(connected=True) #инициализация методов


# In[120]:


rcParams['figure.figsize'] = 8, 6
genre_vs_budget1 = one_genre.loc[:, ['genres_per_one', 'budget', 'year']].groupby(by=['genres_per_one'])[['budget']].sum()
#genre_vs_budget2 = one_genre.loc[:, ['genres_per_one', 'budget']].groupby(by=['genres_per_one']).sum()
#data = genre_vs_budget2.join(genre_vs_budget1)
genre_vs_budget1.plot(kind='bar', rot=45, label='year') 


# In[144]:


movie['original_language'].value_counts()[:10].plot(kind='bar', rot=45)


# In[145]:


d = movie['original_language'].value_counts()/movie.shape[0]*100
d[:10].plot(kind='bar', rot=45)


# In[160]:


d[1:10].plot(kind='bar', rot=45)


# In[155]:


con = (movie.revenue == movie.revenue) & (movie.revenue != 0)
sns.distplot(movie[con].revenue)


# In[148]:


cols = ['budget', 'genres_per_one', 'year', 'revenue', 'original_language']
sns_plot = sns.pairplot(movie[cols])


# In[92]:


genre_vs_budget


# In[156]:


all_genres = set(ll)
all_genres


# In[154]:


movie.genres_per_one.unique()


# In[ ]:


one_genre.groupby(by=['genres_per_one']).adult.count().plot(kind='bar', rot=45)


# In[219]:


ratings_small

