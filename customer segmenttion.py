#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation Project
# 
# ## Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing data

# In[2]:


df=pd.read_csv("C:/Users/rajapriya/OneDrive/Desktop/ML/Mall_Customers.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.dtypes


# In[7]:


df.isnull().sum()


# In[8]:


df.tail()


# ## EDA

# In[9]:


plt.figure(1,figsize=(16, 6))
n=0
for x in ['Age','Annual Income (k$)','Spending Score (1-100)']:
     n+=1
     plt.subplot(1,3,n)
     plt.subplots_adjust(hspace=0.5,wspace=0.5)
     sns.distplot(df[x],bins=20)
     plt.title("Distplot of {}".format(x))
plt.show()


# In[10]:


plt.figure(figsize=(15,6))
sns.countplot(x='Gender',data=df)
plt.show()


# In[11]:


plt.figure(1,figsize=(16, 7))
n=0
for cols in ['Age','Annual Income (k$)','Spending Score (1-100)']:
     n+=1
     plt.subplot(1,3,n)
     sns.set(style="whitegrid")
     plt.subplots_adjust(hspace=0.5,wspace=0.5)
     sns.violinplot(x=cols,y='Gender',data=df)
     plt.ylabel('Gender' if n==1 else '' )
     plt.title("violinplot of {}")
plt.show()


# In[12]:


age_18_25 = df.Age[(df.Age>=18)&(df.Age<=25)]
age_26_35 = df.Age[(df.Age>=26)&(df.Age<=35)]
age_36_45 = df.Age[(df.Age>=36)&(df.Age<=45)]
age_46_55 = df.Age[(df.Age>=46)&(df.Age<=55)]
age_55_above = df.Age[df.Age>=56]
agex = ["18-25","26-35","36-45","46-55","55+"]
agey=[len(age_18_25.values),len(age_26_35.values),len(age_36_45.values),len(age_46_55.values),len(age_55_above.values)]
plt.figure(figsize=(15,6))
sns.barplot(x=agex,y=agey,palette="mako")
plt.xlabel("age")
plt.ylabel("number of customers")
plt.title("Age and Number of customers")
plt.show()


# In[13]:


sns.relplot(x='Annual Income (k$)',y='Spending Score (1-100)',data=df)


# In[14]:


ss_1_20 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=1)&(df["Spending Score (1-100)"]<=20)]
ss_21_40 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=21)&(df["Spending Score (1-100)"]<=40)]
ss_41_60 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=41)&(df["Spending Score (1-100)"]<=60)]
ss_61_80 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=61)&(df["Spending Score (1-100)"]<=80)]
ss_81_100 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=81)&(df["Spending Score (1-100)"]<=100)]

ssx = ["1-20","21-40","41-60","61-80","81-100"]
ssy=[len(ss_1_20.values),len(ss_21_40.values),len(ss_41_60.values),len(ss_61_80.values),len(ss_81_100.values)]
plt.figure(figsize=(15,6))
sns.barplot(x=ssx,y=ssy,palette="rocket")
plt.xlabel("Spending score")
plt.ylabel("number of customers")
plt.title("Spending score and Number of customers")
plt.show()


# In[15]:


ai_0_30 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=0)&(df["Annual Income (k$)"]<=30)]
ai_31_60 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=31)&(df["Annual Income (k$)"]<=60)]
ai_61_90 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=61)&(df["Annual Income (k$)"]<=90)]
ai_91_120 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=91)&(df["Annual Income (k$)"]<=120)]
ai_121_150 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=121)&(df["Annual Income (k$)"]<=150)]

aix = ["0-30000","31000-60000","61000-90000","91000-120000","121000-150000"]
aiy=[len(ai_0_30.values),len(ai_31_60.values),len(ai_61_90.values),len(ai_91_120.values),len(ai_121_150.values)]
plt.figure(figsize=(15,6))
sns.barplot(x=aix,y=aiy,palette="Spectral")
plt.xlabel("IncomeS")
plt.ylabel("number of customers")
plt.title("Annual incomes")
plt.show()


# # Kmeans 
# ## Age and spending score

# In[16]:


X1=df.loc[:,['Age','Spending Score (1-100)']].values
from sklearn.cluster import KMeans
wcss=[]
for k in range(1,11):
    km=KMeans(n_clusters=k,init ="k-means++")
    km.fit(X1)
    wcss.append(km.inertia_)
    plt.figure(figsize=(12,6))

plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()


# In[17]:


km=KMeans(n_clusters=4)
label = km.fit_predict(X1)
print(label)


# In[18]:


print(km.cluster_centers_)


# In[19]:


plt.scatter(X1[:,0],X1[:,1],c=km.labels_,cmap="rainbow")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color="black")
plt.title("clusters of customers")
plt.xlabel('age')
plt.ylabel('spending score (1-100)')
plt.show()


# ## annual income and spending score

# In[20]:


X2=df.loc[:,['Annual Income (k$)','Spending Score (1-100)']].values
from sklearn.cluster import KMeans
wcss=[]
for k in range(1,11):
    km=KMeans(n_clusters=k,init ="k-means++")
    km.fit(X2)
    wcss.append(km.inertia_)
    plt.figure(figsize=(12,6))

plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()


# In[21]:


km=KMeans(n_clusters=5)
label = km.fit_predict(X2)
print(label)


# In[22]:


print(km.cluster_centers_)


# In[23]:


plt.scatter(X2[:,0],X2[:,1],c=km.labels_,cmap="rainbow")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color="black")
plt.title("clusters of customers")
plt.xlabel('Annual income')
plt.ylabel('spending score (1-100)')
plt.show()


# ## 3-Dimensional

# In[24]:


X3=df[["Age","Annual Income (k$)","Spending Score (1-100)"]]
wcss=[]
for k in range(1,11):
    km=KMeans(n_clusters=k,init ="k-means++")
    km.fit(X3)
    wcss.append(km.inertia_)
    plt.figure(figsize=(12,6))

plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()


# In[25]:


km=KMeans(n_clusters=6)
label = km.fit_predict(X3)
print(label)


# In[26]:


print(km.cluster_centers_)


# In[27]:


clusters =  km.fit_predict(X3)
df["label"] = clusters
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.label == 0], df["Annual Income (k$)"][df.label == 0], df["Spending Score (1-100)"][df.label == 0], c='purple', s=60)
ax.scatter(df.Age[df.label == 1], df["Annual Income (k$)"][df.label == 1], df["Spending Score (1-100)"][df.label == 1], c='red', s=60)
ax.scatter(df.Age[df.label == 2], df["Annual Income (k$)"][df.label == 2], df["Spending Score (1-100)"][df.label == 2], c='blue', s=60)
ax.scatter(df.Age[df.label == 3], df["Annual Income (k$)"][df.label == 3], df["Spending Score (1-100)"][df.label == 3], c='green', s=60)
ax.scatter(df.Age[df.label == 4], df["Annual Income (k$)"][df.label == 4], df["Spending Score (1-100)"][df.label == 4], c='yellow', s=60)
ax.scatter(df.Age[df.label == 5], df["Annual Income (k$)"][df.label == 5], df["Spending Score (1-100)"][df.label == 5], c='brown', s=60)
ax.view_init(35, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()


# In[28]:


cust1=df[df["label"]==1]
print('Number of customer in 1st group=', len(cust1))
print('They are -', cust1["CustomerID"].values)
print("--------------------------------------------")
cust2=df[df["label"]==2]
print('Number of customer in 2nd group=', len(cust2))
print('They are -', cust2["CustomerID"].values)
print("--------------------------------------------")
cust3=df[df["label"]==0]
print('Number of customer in 3rd group=', len(cust3))
print('They are -', cust3["CustomerID"].values)
print("--------------------------------------------")
cust4=df[df["label"]==3]
print('Number of customer in 4th group=', len(cust4))
print('They are -', cust4["CustomerID"].values)
print("--------------------------------------------")
cust5=df[df["label"]==4]
print('Number of customer in 5th group=', len(cust5))
print('They are -', cust5["CustomerID"].values)
print("--------------------------------------------")
cust6=df[df["label"]==5]
print('Number of customer in 6th group=', len(cust6))
print('They are -', cust6["CustomerID"].values)
print("--------------------------------------------")


# In[ ]:





# In[ ]:




