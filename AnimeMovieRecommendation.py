#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install numpy')


# In[2]:


get_ipython().system('pip3 install pandas')


# In[3]:


get_ipython().system('pip3 install matplotlib')


# In[4]:


get_ipython().system('pip3 install scipy')


# In[5]:


get_ipython().system('pip3 install helper')


# In[6]:


get_ipython().system('pip3 install sklearn')


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import itertools
from sklearn.metrics import silhouette_samples, silhouette_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


anime = pd.read_csv('../anime.csv')


# In[9]:


anime.head(10)


# In[10]:


print(anime.shape)


# In[11]:


ratings = pd.read_csv('../rating.csv')


# In[12]:


ratings.head(10)


# In[13]:


# Print the number of records and the total number of movies
print('The dataset contains: ', len(ratings), ' ratings of ', len(anime), ' movies.')


# In[14]:


def get_genre_ratings(ratings, anime, genre, column_names):
    genre_ratings = pd.DataFrame()
    for genres in genre:        
        genre_movies = anime[anime['genre'].str.contains(genres, case=False, na=False) ]
        avg_genre_votes_per_user = ratings[ratings['anime_id'].isin(genre_movies['anime_id'])].loc[:, ['user_id', 'rating']].groupby(['user_id'])['rating'].mean().round(2)
        
        genre_ratings = pd.concat([genre_ratings, avg_genre_votes_per_user], axis=1)
        
    genre_ratings.columns = column_names
    return genre_ratings

genre_ratings = get_genre_ratings(ratings, anime, ['Romance','Drama'], ['avg_romance_rating', 'avg_drama_rating'])


# In[15]:


print('The dataset contains: ', len(ratings), ' ratings of ', len(anime), 'anime movies.')


# In[16]:


genre_ratings.head()


# In[17]:


# Function to get the biased dataset
def bias_genre_rating_dataset(genre_ratings, score_limit_1, score_limit_2):
    biased_dataset =  genre_ratings[((genre_ratings['avg_romance_rating'] < score_limit_1 - 0.2) & (genre_ratings['avg_drama_rating'] > score_limit_2)) | ((genre_ratings['avg_romance_rating'] < score_limit_1) & (genre_ratings['avg_drama_rating'] > score_limit_2))]
    biased_dataset = pd.concat([biased_dataset[:300], genre_ratings[:2]])
    biased_dataset = pd.DataFrame(biased_dataset.to_records())    
    return biased_dataset# Bias the dataset
biased_dataset = bias_genre_rating_dataset(genre_ratings, 3.0, 2.8)# Printing the resulting number of records & the head of the dataset
print( "Number of records: ", len(biased_dataset))


# In[18]:


biased_dataset.head()


# In[19]:


# Defining the scatterplot drawing function
def draw_scatterplot(x_data, x_label, y_data, y_label):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)    
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.scatter(x_data, y_data, s=30)


# In[20]:


# Plot the scatterplot
draw_scatterplot(biased_dataset['avg_romance_rating'],'Avg Romance rating', biased_dataset['avg_drama_rating'], 'Avg drama rating')


# In[24]:


ratings[ratings['user_id']==1].rating.mean()


# In[25]:


# Let's turn our dataset into a list
X = biased_dataset[['avg_romance_rating','avg_drama_rating']].values# Import KMeans
from sklearn.cluster import KMeans# Create an instance of KMeans to find two clusters
kmeans_1 = KMeans(n_clusters=2)# Use fit_predict to cluster the dataset
predictions = kmeans_1.fit_predict(X)# Defining the cluster plotting function
def draw_clusters(biased_dataset, predictions, cmap='viridis'):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    ax.set_xlabel('Avg romance rating')
    ax.set_ylabel('Avg drama rating')
    clustered = pd.concat([biased_dataset.reset_index(), pd.DataFrame({'group':predictions})], axis=1)
    plt.scatter(clustered['avg_romance_rating'], clustered['avg_drama_rating'], c=clustered['group'], s=20, cmap=cmap)# Plot
draw_clusters(biased_dataset, predictions)


# In[26]:


# Create an instance of KMeans to find three clusters
kmeans_2 = KMeans(n_clusters=3)# Use fit_predict to cluster the dataset
predictions_2 = kmeans_2.fit_predict(X)# Plot
draw_clusters(biased_dataset, predictions_2)


# In[27]:


# Create an instance of KMeans to find three clusters
kmeans_3 = KMeans(n_clusters=4)# Use fit_predict to cluster the dataset
predictions_3 = kmeans_3.fit_predict(X)# Plot
draw_clusters(biased_dataset, predictions_3)


# In[35]:


#   In this kernel, I decide to reduce size of dataset, because of running time
mergedata = pd.merge(anime,ratings,on=['anime_id','anime_id'])
mergedata= mergedata[mergedata.user_id <= 20000]
mergedata.head(10)


# In[36]:


#Show detail of anime which each user like
user_anime = pd.crosstab(mergedata['user_id'], mergedata['name'])
user_anime.head(10)


# In[37]:


#Principal Component Analysis converts our original variables to a new set of variables, which are a linear combination of the original set of variables.My main goal is to reduce dimension of data for clustering and visualize
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(user_anime)
pca_samples = pca.transform(user_anime)



# In[38]:


ps = pd.DataFrame(pca_samples)
ps.head()



# In[39]:


tocluster = pd.DataFrame(ps[[0,1,2]])


# In[40]:


from mpl_toolkits.mplot3d import axes3d    

plt.rcParams['figure.figsize'] = (16, 9)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tocluster[0], tocluster[2], tocluster[1])

plt.title('Data points in 3D PCA axis', fontsize=20)
plt.show()


# In[41]:


#Selecting number of k

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

scores = []
inertia_list = np.empty(8)

for i in range(2,8):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(tocluster)
    inertia_list[i] = kmeans.inertia_
    scores.append(silhouette_score(tocluster, kmeans.labels_))
    


# In[42]:


plt.plot(range(0,8),inertia_list,'-o')
plt.xlabel('Number of cluster')
plt.axvline(x=4, color='blue', linestyle='--')
plt.ylabel('Inertia')
plt.show()


# In[43]:


plt.plot(range(2,8), scores);
plt.title('Results KMeans')
plt.xlabel('n_clusters');
plt.axvline(x=4, color='blue', linestyle='--')
plt.ylabel('Silhouette Score');
plt.show()


# In[44]:


#K means clustering
from sklearn.cluster import KMeans

clusterer = KMeans(n_clusters=4,random_state=30).fit(tocluster)
centers = clusterer.cluster_centers_
c_preds = clusterer.predict(tocluster)

print(centers)


# In[45]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tocluster[0], tocluster[2], tocluster[1], c = c_preds)
plt.title('Data points in 3D PCA axis', fontsize=20)

plt.show()


# In[46]:


fig = plt.figure(figsize=(10,8))
plt.scatter(tocluster[1],tocluster[0],c = c_preds)
for ci,c in enumerate(centers):
    plt.plot(c[1], c[0], 'o', markersize=8, color='red', alpha=1)

plt.xlabel('x_values')
plt.ylabel('y_values')

plt.title('Data points in 2D PCA axis', fontsize=20)
plt.show()


# In[47]:


user_anime['cluster'] = c_preds


user_anime.head(10)


# In[48]:


#characteristic of each cluster

c0 = user_anime[user_anime['cluster']==0].drop('cluster',axis=1).mean()
c1 = user_anime[user_anime['cluster']==1].drop('cluster',axis=1).mean()
c2 = user_anime[user_anime['cluster']==2].drop('cluster',axis=1).mean()
c3 = user_anime[user_anime['cluster']==3].drop('cluster',axis=1).mean()


# In[49]:


#cluster 0
c0.sort_values(ascending=False)[0:15]


# In[62]:


#Average of each information for anime which user in this cluster like

print('cluster 0\nAVG episode : {0}\nAVG movie rating : {1}\nAVG member : {2}'
      .format(c0_data['episode'].mean(), c0_data['rating'].mean(),c0_data['member'].mean()))


# In[63]:


#cluster 1

c1.sort_values(ascending=False)[0:15]


# In[65]:


#Average of each information for anime which user in this cluster like



print('cluster 1\nAVG episode : {0}\nAVG movie rating : {1}\nAVG member : {2}'
      .format(c1_data['episode'].mean(), c1_data['rating'].mean(),c1_data['member'].mean()))



# In[66]:


#cluster 2
c2.sort_values(ascending=False)[0:15]


# In[69]:


#Average of each information for anime which user in this cluster like


print('cluster 2\nAVG episode : {0}\nAVG movie rating : {1}\nAVG member : {2}'
      .format(c2_data['episode'].mean(), c2_data['rating'].mean(),c2_data['member'].mean()))



# In[70]:


#cluster 3
c3.sort_values(ascending=False)[0:15]


# In[73]:


#Average of each information for anime which user in this cluster like


print('cluster 3\nAVG episode : {0}\nAVG movie rating : {1}\nAVG member : {2}'
      .format(c3_data['episode'].mean(), c3_data['rating'].mean(),c3_data['member'].mean()))


# In[ ]:




