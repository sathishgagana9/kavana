#!/usr/bin/env python
# coding: utf-8

# In[331]:


import pandas as pd 
import numpy as np 
data={'english':[85,56,np.nan,95,85,56],
     'kannada':[56,75,89,np.nan,85,97],
     'pysics':[85,75,np.nan,84,89,90],
     'chemistry':[98,78,79,58,np.nan,89],
     'biology':[85,95,68,48,np.nan,30]}
df=pd.DataFrame(data)
df


# In[332]:


df['total']=df['english']+df['kannada']+df['pysics']+df['chemistry']+df['biology']


# In[333]:


df


# In[334]:


df.isnull()


# In[335]:


df.notnull()


# In[336]:


df.fillna(df.mean())


# In[337]:


df.bfill()


# In[338]:


df.ffill()


# In[339]:


df.replace()


# In[340]:


df


# In[341]:


import matplotlib.pyplot as plt
import missingno as msn
msn.bar(df)
plt.show()


# In[342]:


df.interpolate()


# In[343]:


df


# In[344]:


df.fillna(60,inplace=True)


# In[345]:


df+5


# In[346]:


df=df+[4,5,2,1,3,0]
df


# In[347]:


df['english'] > 70


# In[348]:


df[df['total']>200]


# In[349]:


df[df['total']<200]


# In[350]:


df.loc[6]=[56,85,85,75,65,98]


# In[351]:


df


# In[352]:


df['english'].mean()


# In[353]:


high=df[df['total']==df['total'].max()]
low=df[df['total']==df['total'].min()]
print(high,"\n")
print(low)


# In[354]:


df.apply(np.sum,axis=1)


# In[355]:


df.apply(np.sum,axis=0)


# In[356]:


import pandas as pd 
import numpy as np 
data={'ename':['gagana','chethu','bhoomi','teju','ammu','pratti'],
     'department':['cs','cs','ec','cs','ec','ec'],
     'experiance':[6,5,2,3,1,2],
     'salary':[100000,52000,40000,30000,45000,70000]}
df=pd.DataFrame(data)
df


# In[357]:


df['id']=[111,222,333,444,555,666]


# In[358]:


df


# In[359]:


df.groupby('department')['salary'].sum()


# In[360]:


df.groupby('department')['salary'].mean()


# In[361]:


avg=df.pivot_table(values='salary',index='department',columns='experiance',aggfunc='mean')
avg


# In[362]:


sum=df.pivot_table(values='salary',index='department',columns='experiance',aggfunc='sum')
sum


# In[363]:


df['department'].value_counts()


# In[364]:


df['department'].values


# In[365]:


import numpy as np 
import pandas as pd 
data=pd.DataFrame({'value':[1,2,3,4,5,15,5,15,42,100,4,26,23,23]})
mean=data['value'].mean()
std=data['value'].std()
print(mean)
print(std)
threshold=2
outlier=[]
for i in data['value']:
    z=(i-mean)/std
    if z > threshold:
        outlier.append(i)
print("outlier",outlier)


# In[366]:


data=pd.DataFrame({'value':[1,2,3,4,5,15,5,15,42,100,4,26,23,23]})
q1=data['value'].quantile(0.25)
q3=data['value'].quantile(0.75)
iqr=q3-q1
lower_bound=q1-1.5*iqr
upper_bound=q3+1.5*iqr
outlier=data[(data['value']<lower_bound)|(data['value']>upper_bound)]
print(outlier)


# In[367]:


plt.scatter(data['value'],data.index)
plt.show()


# In[368]:


plt.bar(data['value'],data.index)
plt.show()


# In[369]:


import seaborn as sns
sns.histplot(data['value'])
plt.show()


# In[370]:


plt.boxplot(data['value'])
plt.show()


# In[371]:


m=data['value'].mean()
for i in data['value']:
    if i < lower_bound or i > upper_bound:
        data['value']=data['value'].replace(i,m)


# In[372]:


data


# In[412]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error,classification_report,confusion_matrix


# In[413]:


df=pd.read_csv("auto-mpg.csv")
df


# In[414]:


le=LabelEncoder()
df['car name']=le.fit_transform(df['car name'])
df


# In[415]:


x=df.drop('car name',axis=1)
y=df['car name']


# In[416]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=110)
x_train


# In[417]:


#df[df['cylinders']==8]


# In[418]:


scaler=StandardScaler()
scaler.fit(x_test)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
print(x_train)
print(x_test)


# In[419]:


le=LinearRegression()
model=le.fit(x_train,y_train)
model


# In[421]:


y_pred=model.predict(x_test)


# In[422]:


abc=pd.DataFrame({'y_pred':y_pred,'y_test':y_test})
abc


# In[424]:


mse=mean_squared_error(y_test,y_pred)
mse


# In[426]:


from sklearn.linear_model import LogisticRegression
le=LogisticRegression()
model1=le.fit(x_train,y_train)
model1


# In[428]:


y_pred=model1.predict(x_test)
abc=pd.DataFrame({'y_pred':y_pred,'y_test':y_test})
abc


# In[429]:


acc=accuracy_score(y_test,y_pred)
acc


# In[436]:


from sklearn.tree import DecisionTreeClassifier
de=DecisionTreeClassifier()
model2=de.fit(x_train,y_train)
model2


# In[437]:


y_pred=model2.predict(x_test)
abc=pd.DataFrame({'y_pred':y_pred,'y_test':y_test})
abc


# In[438]:


acc=accuracy_score(y_test,y_pred)
acc


# In[440]:


from sklearn.tree import DecisionTreeRegressor
de=DecisionTreeRegressor()
model3=de.fit(x_train,y_train)
model3


# In[441]:


y_pred=model3.predict(x_test)
abc=pd.DataFrame({'y_pred':y_pred,'y_test':y_test})
abc


# In[443]:


acc=r2_score(y_test,y_pred)
acc


# In[444]:


from sklearn.ensemble import RandomForestRegressor
de=RandomForestRegressor()
model4=de.fit(x_train,y_train)
model4


# In[445]:


y_pred=model4.predict(x_test)
abc=pd.DataFrame({'y_pred':y_pred,'y_test':y_test})
abc


# In[447]:


acc=r2_score(y_test,y_pred)
acc


# In[448]:


from sklearn.ensemble import RandomForestClassifier
de=RandomForestClassifier()
model5=de.fit(x_train,y_train)
model5


# In[449]:


y_pred=model5.predict(x_test)
abc=pd.DataFrame({'y_pred':y_pred,'y_test':y_test})
abc


# In[451]:


acc=accuracy_score(y_test,y_pred)
acc


# In[455]:


from sklearn.svm import SVC
de=SVC(kernel='linear',gamma=0.5)
model6=de.fit(x_train,y_train)
model6


# In[456]:


y_pred=model6.predict(x_test)
abc=pd.DataFrame({'y_pred':y_pred,'y_test':y_test})
abc


# In[457]:


acc=accuracy_score(y_test,y_pred)
acc


# In[462]:


from sklearn.neighbors import KNeighborsClassifier
de=KNeighborsClassifier()
model7=de.fit(x_train,y_train)
model7


# In[463]:


y_pred=model7.predict(x_test)
abc=pd.DataFrame({'y_pred':y_pred,'y_test':y_test})
abc


# In[465]:


acc=accuracy_score(y_test,y_pred)
acc


# In[ ]:


from sklearn.cluster import KMeans
wcss=[]


# In[ ]:




