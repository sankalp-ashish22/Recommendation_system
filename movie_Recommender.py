#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import pandas as pd 


# In[4]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[5]:


movies.head(1)


# In[6]:


movies=movies.merge(credits,on="title")


# In[7]:


credits.shape


# In[8]:


movies.shape


# In[9]:


features = ['genres', 'movie_id','keywords','title','overview','cast','crew']


# In[10]:


movies = movies[features]


# In[11]:


movies.head()


# In[12]:


movies.isnull().sum()


# In[13]:


movies.dropna(inplace=True)


# In[14]:


movies.isnull().sum()


# In[15]:


movies.duplicated().sum()


# In[16]:


movies.iloc[0].genres 


# In[17]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[18]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[19]:


movies['genres']=movies['genres'].apply(convert)


# In[20]:


movies.head()


# In[21]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[22]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L 


# In[23]:


movies['cast'].apply(convert3)


# In[24]:


movies['cast'] = movies['cast'].apply(convert3)


# In[25]:


movies.head()


# In[26]:


movies['crew'][0]


# In[27]:


def director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
    return L        


# In[28]:


movies['crew'].apply(director)


# In[29]:


movies['crew'] = movies['crew'].apply(director)


# In[30]:


movies.head()


# In[31]:


movies['overview'][0]


# In[32]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[33]:


movies.head()


# In[34]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])


# In[35]:


movies.head()


# In[36]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[37]:


movies.head()


# In[38]:


new_df = movies[['movie_id','title','tags']]


# In[39]:


new_df


# In[40]:


new_df['tags'].apply(lambda x:" ".join(x))


# In[41]:


new_df.loc[:, 'tags']= new_df['tags'].apply(lambda x:" ".join(x))


# In[42]:


new_df['tags'][0]



# In[43]:


new_df.loc[:,'tags']=new_df['tags'].apply(lambda x:x.lower())


# In[44]:


new_df.head()


# In[45]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[46]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[47]:


import nltk 


# In[48]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[49]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[50]:


# get_ipython().system('pip install nltk')


# In[51]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')


# In[52]:


new_df['tags'].apply(stem)


# In[53]:


new_df.loc[:,'tags'] = new_df['tags'].apply(stem)


# In[55]:


vectors


# In[56]:


from sklearn.metrics.pairwise import cosine_similarity


# In[57]:


similarity=cosine_similarity(vectors)


# In[58]:


similarity[1]


# In[59]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[60]:


def recommend(movie):
    names=[]
    movie_index = new_df[new_df['title']==movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movie_list:
        print(new_df.iloc[i[0]].title)
        names.append(new_df.iloc[i[0]].title)
    return names
            
              
# In[62]:


recommend('Avatar')


# In[ ]:





# In[63]:




