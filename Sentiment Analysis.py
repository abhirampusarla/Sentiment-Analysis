#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

plt.style.use('ggplot')
# reading file from csv
df = pd.read_csv("*Add your Path* /Reviews.csv")
print(df.shape)
df = df.head(500)
print(df.shape)
df.head()


# In[28]:


#This is important for viewing customer review

df['Text'].values[4]


# In[34]:


# Defines count of each star_ratings  

ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of each star_ratings',
          figsize=(10, 5))
ax.set_xlabel('Number of Stars')
ax.set_ylabel('Number of people')
plt.show()


# In[47]:


example = df['Text'][50]
print(example)

tokens = nltk.word_tokenize(example)
tokens[:10]
tagged = nltk.pos_tag(tokens)
tagged[:10]

# Creating chunk
entities = nltk.chunk.ne_chunk(tagged)
print(entities)


# In[53]:


# Sentiment Analysis

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
sia = SentimentIntensityAnalyzer()
sia.polarity_scores('I am so happy!')


# In[55]:


sia.polarity_scores("worst thing happened ever")


# In[57]:


sia.polarity_scores(example)


# In[58]:


# Running polarity score on entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)


# In[62]:


res


# In[67]:


vaders = pd.DataFrame(res).T

vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')


# In[68]:


vaders


# In[69]:


# Plotting Results
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compund Score by Amazon Star Review')
plt.show()


# In[91]:


#Plotting Number of Positive review, Negative Review, Neutral reviews 

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

