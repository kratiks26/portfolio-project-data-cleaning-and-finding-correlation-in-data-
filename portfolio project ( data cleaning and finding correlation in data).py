#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing liberaries

import numpy as np
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure


# In[3]:


# importing data 
df=pd.read_csv(r'C:\Users\krati\Documents\Portfolio projects\pyhton project\movies\movies.csv')


# In[4]:


df.head()


# # Data cleaning

# In[5]:


# lets just find the percentage or null vales in Data

for col in df.columns:
    per_null=np.mean(df[col].isnull())
    print('{}- {}%'.format(col,per_null))


# In[6]:


# delete all the columns having null values
df=df.dropna()


# In[7]:


# there is no null values
for col in df.columns:
    per_null=np.mean(df[col].isnull())
    print('{}- {}%'.format(col,per_null))


# In[8]:


# changing all unnecessory float columns to int
df.dtypes


# In[9]:


df['score']=df['score'].astype('category')


# In[10]:


df
for col in df.columns:
    if(df[col].dtype == 'float64'):
        df[col]=df[col].astype('int64')


# In[11]:


df['score']=df['score'].astype('float64')


# In[12]:


df.dtypes


# In[13]:


# we have wrong and unwanted mentioned Conutry name in released column, so we have to delete it
df.head()


# In[14]:


df1=pd.DataFrame(df['released'].str.split('(', expand=True))
df1.head()


# In[15]:


df['released_new']=df1[0]


# In[16]:


del(df['released'])


# In[17]:


# now rename newly created column
df=df.rename(columns={'released_new':'released'})


# In[18]:


pd.set_option('display.max_rows',None)


# In[19]:


df.head()


# In[20]:


# now we can see we have different years in released date column and year
# we considered released date colun se correct and change year according to..
df2= pd.DataFrame(df['released'].str.split(',', expand=True))
df2.head()


# In[23]:


del(df['year'])


# In[24]:


df['year']=df2[1]


# In[25]:


df.head()


# In[26]:


# sort dataframe by desc year to see if is there any null value
df.sort_values(by='year', ascending=False).tail(16)


# In[27]:


# we have 14 nul values in last, lets fill it manually
df.loc[312,'year']=1982
df.loc[449,'year']=1983
df.loc[467,'year']=1985
df.loc[800,'year']=1985
df.loc[1173,'year']=1987
df.loc[1212,'year']=1988
df.loc[1404,'year']=1988
df.loc[1819,'year']=1991
df.loc[2029,'year']=1991
df.loc[2318,'year']=1994
df.loc[2319,'year']=1994
df.loc[2816,'year']=1995
df.loc[4187,'year']=2019
df.loc[5833,'year']=2010


# In[28]:


df['year']=df['year'].astype('int64')


# In[29]:


# now we can see there is no null values
for col in df.columns:
    per_null=np.mean(df[col].isnull())
    print('{}- {}%'.format(col,per_null))


# ### now our DataFrme is all filled and Considered as Correct

# In[30]:


pd.reset_option('display.max_rows',None)


# In[31]:


# now lets find the correlation between the numeric values of the data 
df.corr()


# In[33]:


# lets make a heat map for batter visulation
corr_matrix=df.corr()
plt.figure(figsize=(18,9), dpi=150)
sns.heatmap(corr_matrix,annot=True)


# In[34]:


# lets make a scatter plot budget vs gross revenue
plt.figure(figsize=(18,9), dpi=150)
plt.scatter(x=df['gross'], y=df['budget'], color='m')
plt.title('Correlation plot Between Budget and Gross Revenue', size=20 )
plt.grid(True)


# In[35]:


# now lets make a regression line in this plot 
plt.figure(figsize=(18,9), dpi=150)
sns.regplot(x='gross', y='budget', data=df, scatter_kws={'color':'m'}, line_kws={'color':'blue'})
plt.grid(True)


# ## we can even find the correlation between object and numeric data, For that we need to convert object dtypes as code which will also be the nemuric representation of data

# In[36]:


# first let's just copy our DataFrame so it will not effect the actuall DataFrame
tdf=df.copy()


# In[37]:


tdf
for col in tdf.columns:
    if(tdf[col].dtypes == 'object'):
        tdf[col]=tdf[col].astype('category')
        tdf[col]=tdf[col].cat.codes


# In[38]:


tdf.head()
# now we can see that every data is converted into numbers


# In[39]:


# now we can make a correlation matrix between two data.
tdf.corr()


# In[40]:


tdf_corr_matrix=tdf.corr()
plt.figure(figsize=(18,9),dpi=150)
sns.heatmap(tdf_corr_matrix, annot=True)


# ## So Here We Can Easly See That There Is A High Correlation Between:- 
# ## Gross Revenue and Budget,
# ## Gross Revenue and Vote,
# # and We Can Even Consider Votes and Score as Correlated

# In[ ]:




