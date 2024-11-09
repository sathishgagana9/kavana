#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
data={'english':[85,56,75,95,85,56],
     'kannada':[56,75,89,68,85,97],
     'pysics':[85,75,78,84,89,90],
     'chemistry':[98,78,79,58,68,89],
     'biology':[85,95,68,48,np.nan,30]}
df=pd.DataFrame(data)
df


# In[3]:


df


# In[4]:


df['maths']=[85,75,59,68,79,80]


# In[5]:


df


# In[6]:


df['total']=df['english']+df['kannada']+df['pysics']+df['chemistry']+df['biology']+df['maths']


# In[7]:


df


# In[8]:


df+5


# In[9]:


df-2


# In[10]:


df['english']+3


# In[11]:


df.columns


# In[12]:


df.dtypes


# In[13]:


df


# In[14]:


df.drop('total',axis=1,inplace=True)


# In[15]:


df


# In[16]:


df['english'].mean()


# In[17]:


df['kannada'].max()


# In[18]:


df['pysics'].min()


# In[19]:


df['kannada'].std()


# In[20]:


df['kannada'].median()


# In[21]:


df.apply(np.sum,axis=0)


# In[22]:


df.apply(np.sum,axis=1)


# In[23]:


high=df[df['maths']==df['maths'].max()]
low=df[df['maths']==df['maths'].min()]
print(high)
print(low)


# In[24]:


import pandas as pd 
import numpy as np 
data={'ename':['gagana','chethu','bhoomi','teju','ammu','pratti'],
     'department':['cs','cs','ec','cs','ec','ec'],
     'experiance':[6,5,2,3,1,2],
     'salary':[100000,52000,40000,30000,45000,70000]}
df=pd.DataFrame(data)


# In[25]:


df


# In[26]:


df['id']=[111,222,333,444,555,666]
df


# In[27]:


avg=df.pivot_table(values='salary',index='department',columns='experiance',aggfunc='mean')
avg


# In[28]:


sum=df.pivot_table(values='salary',index='department',columns='experiance',aggfunc=['mean','sum'])
sum


# In[29]:


df['department'].value_counts()


# In[30]:


df['experiance'].value_counts()


# In[31]:


ab=df.groupby('ename')['salary'].mean()
ab


# In[32]:


ab=df.groupby('experiance')['salary'].mean()
ab


# In[33]:


ab=df.groupby('department')['salary'].mean()
ab


# In[34]:


df['age']=[45,63,32,67,45,75]


# In[35]:


df


# In[36]:


df.loc[6]=['kavana','cs',3,50000,777,41]


# In[37]:


df


# In[38]:


df[df['age']>40]


# In[39]:


df[df['age']<40]


# In[40]:


df[df['age']==45]


# In[41]:


df[df['ename']=='gagana']


# In[42]:


df.drop('id',axis=1,inplace=True)


# In[43]:


df


# In[44]:


df.drop(index=6,axis=0,inplace=True)


# In[45]:


df


# In[46]:


import pandas as pd 
import numpy as np 
data={'english':[85,56,np.nan,95,85,56],
     'kannada':[np.nan,75,89,68,np.nan,97],
     'pysics':[85,75,np.nan,84,89,90],
     'chemistry':[98,78,79,58,68,np.nan],
     'biology':[85,np.nan,68,48,np.nan,30]}
df=pd.DataFrame(data)
df


# In[47]:


df.fillna(20)


# In[48]:


df.bfill()


# In[49]:


df.ffill()


# In[50]:


df.fillna(df.mean())


# In[51]:


df


# In[52]:


df.loc[2].fillna(20)


# In[53]:


df['kannada'].fillna(20)


# In[54]:


df.isnull()


# In[55]:


df.isnull().sum()


# In[56]:


df.notnull()


# In[57]:


df.replace()


# In[58]:


df.interpolate()


# In[60]:


import missingno as msn
msn.bar(df)


# In[61]:


import pandas as pd 
import numpy as np 
data=pd.DataFrame({'value':[1,2,3,4,5,6,2,1,2,3,15,2,2,3,4,5,6,5]})
mean=data['value'].mean()
std=data['value'].std()
print("mean",mean)
print("std",std)
threshold=2
outlier=[]
for i in data['value']:
    z=(i-mean)/std
    if z > threshold:
        outlier.append(i)
print("oulier",outlier)


# In[62]:


import pandas as pd 
import numpy as np 
data=pd.DataFrame({'value':[1,2,3,4,5,6,2,1,2,3,15,2,2,3,4,5,6,5]})
q1=data['value'].quantile(0.25)
q3=data['value'].quantile(0.75)
iqr=q3-q1
lower_bound=q1-1.5*iqr
upper_bound=q3+1.5*iqr
outlier=data[(data['value'] < lower_bound) | (data['value'] > upper_bound)]
print(outlier)


# In[63]:


import matplotlib.pyplot as plt
plt.bar(data['value'],data.index)
plt.show()


# In[64]:


import matplotlib.pyplot as plt
plt.scatter(data['value'],data.index)
plt.show()


# In[ ]:


plt.boxplot(data['value'])
plt.show()


# In[66]:


import matplotlib.pyplot as plt
plt.hist(data['value'],data.index)
plt.show()


# In[67]:


m=data['value'].mean()
for i in data['value']:
    if i < lower_bound or i > upper_bound:
        data['value']=data['value'].replace(i,m)


# In[68]:


data


# In[69]:


import pandas as pd 
import numpy as np 
data=pd.DataFrame({'value':[1,2,3,4,5,6,2,1,2,3,15,2,2,3,4,5,6,5]})
q1=data['value'].quantile(0.25)
q3=data['value'].quantile(0.75)
iqr=q3-q1
lower_bound=q1-1.5*iqr
upper_bound=q3+1.5*iqr
outlier=data[(data['value'] < lower_bound) | (data['value'] > upper_bound)]
print(outlier)


# In[70]:


m2=data['value'].median()
for i in data['value']:
    if i < lower_bound or i > upper_bound:
        data['value']=data['value'].replace(i,m2)


# In[71]:


data


# In[72]:


for i in data['value']:
    if i < lower_bound or i > upper_bound:
        data['value']=data['value'].replace(i,0)


# In[73]:


df=pd.read_csv("iris.csv")
df


# In[74]:


x=df.iloc[:,0:2].values
print(x.shape)


# In[75]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("the elbow method")
plt.xlabel("no of clusters")
plt.ylabel("index")
plt.show()


# In[76]:


km1=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)


# In[77]:


y_means=km1.fit_predict(x)


# In[78]:


y_means


# In[79]:


km1.cluster_centers_


# In[80]:


plt.scatter(x[y_means==0,0],x[y_means==0,1],s=100,c='pink',label='c1:kanjoos')
plt.scatter(x[y_means==1,0],x[y_means==1,1],s=100,c='orange',label='c1:average')
plt.scatter(x[y_means==2,0],x[y_means==2,1],s=100,c='red',label='c1:target')
plt.scatter(x[y_means==3,0],x[y_means==3,1],s=100,c='yellow',label='c1:pokri')
plt.scatter(x[y_means==4,0],x[y_means==4,1],s=100,c='cyan',label='c1:intelligent')
plt.scatter(km1.cluster_centers_[:,0],km1.cluster_centers_[:,1],s=100,c="blue",label="centroid")

plt.title("cluster")
plt.legend()
plt.show()


# In[81]:


import matplotlib.pyplot as plt
plt.hist(df['class'])
plt.show()


# In[82]:


import matplotlib.pyplot as plt
plt.bar(df['sepallength'],df.index)
plt.show()


# In[83]:


import matplotlib.pyplot as plt
plt.boxplot(df['sepallength'])
plt.show()


# In[84]:


import matplotlib.pyplot as plt
plt.plot(df['class'])
plt.show()


# In[85]:


import matplotlib.pyplot as plt
plt.scatter(df['petallength'],df['petalwidth'])
plt.show()


# In[86]:


import seaborn as sns
sns.histplot(df['class'])
plt.show()


# In[87]:


import seaborn as sns
sns.lineplot(df['sepallength'],label='epallength')
sns.lineplot(df['sepalwidth'],label= 'palwidth')
plt.show()


# In[88]:


import seaborn as sns
sns.boxplot(df['sepalwidth'])
plt.show()


# In[89]:


from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
df['class']=le.fit_transform(df['class'])


# In[90]:


import seaborn as sns
sns.barplot(df['class'])
plt.show()


# In[91]:


import seaborn as sns
sns.countplot(df['class'])
plt.show()


# In[92]:


import seaborn as sns
sns.displot(df['class'])
plt.show()


# In[93]:


import nltk


# In[94]:


file=open("word.txt")
text=file.read()
print(text)


# In[95]:


from nltk.tokenize import sent_tokenize


# In[96]:


sent=sent_tokenize(text)
for i in range(len(sent)):
    print("\nsenetence",i+1,"\n",sent[i])


# In[97]:


from nltk.tokenize import word_tokenize
words=word_tokenize(text)
print(len(words),"\n",words)


# In[98]:


from nltk.probability import FreqDist
all_fdist=FreqDist(words)
all_fdist


# In[99]:


import pandas as pd 
import matplotlib.pyplot as plt
all_fdist=pd.Series(dict(all_fdist))
fig,ax=plt.subplots(figsize=(8,8))
all_fdist.plot(kind='bar')
plt.title("frequncy dist of words")
plt.xlabel(words)
plt.show()


# In[100]:


import nltk 
words=word_tokenize(text)
stopwords=nltk.corpus.stopwords.words('english')
remove=[]
for word in words:
    if word in stopwords:
        pass
    else:
        remove.append(word)
        
all_fdist=FreqDist(remove).most_common(10)
all_fdist=pd.Series(dict(all_fdist))
fig,ax=plt.subplots(figsize=(8,8))
all_fdist.plot(kind='bar')
plt.title("frequncy dist of words")
plt.xlabel(words)
plt.show()


# In[101]:


from wordcloud import WordCloud,STOPWORDS
stopwords=set(STOPWORDS)
wordcloud=WordCloud(width=800,height=800,background_color='white',stopwords=stopwords,min_font_size=10).generate(text)
plt.tight_layout(pad=0)
plt.axis('off')
plt.imshow(wordcloud)
plt.show()


# In[102]:


from skimage.io import imread
cloud=imread("cloud.png")
plt.imshow(cloud)
plt.show()


# In[103]:


wordcloud=WordCloud(width=800,height=800,background_color='white',stopwords=stopwords,mask=cloud,min_font_size=10).generate(text)
plt.tight_layout(pad=0)
plt.axis('off')
plt.imshow(wordcloud)
plt.show()


# In[104]:


from nltk.metrics.distance import edit_distance


# In[105]:


from nltk.corpus import words
correct_words=words.words()


# In[106]:


incorrect_words=["appple","amazaing","intelligetn"]
for words in incorrect_words:
    temp=[(edit_distance(words,w),w) for w in correct_words if w[0]==words[0]]
    print(sorted(temp,key=lambda val:val[0])[0][1])


# In[107]:


from nltk.stem import PorterStemmer
ps=PorterStemmer()
sem=[ps.stem(words_sent) for words_sent in words]
sem


# In[108]:


from nltk.stem.wordnet import WordNetLemmatizer
ps=WordNetLemmatizer()
sem=[ps.lemmatize(words_sent) for words_sent in words]
sem


# In[109]:


words=word_tokenize(text)
print(len(words),"\n",nltk.pos_tag(words))


# In[110]:


text=text.lower


# In[111]:


import re
text=re.sub('[^A-Za-z0-9]+',' ',text)
text=re.sub("\S*\d\S*"," ",text).strip()


# In[ ]:




