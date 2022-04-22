#!/usr/bin/env python
# coding: utf-8

# ### Import library

# In[1]:


from __future__ import division

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('pylab', 'inline')
import datetime as dt
import calendar
from haversine import haversine


# ### Import data

# In[2]:


df = pd.read_csv("201508-citibike-tripdata.csv")
df.head()


# ### Answer

# In[3]:


df.shape


# In[4]:


df.sample(5)


# In[5]:


df.dtypes


# #### Data Prep

# In[6]:


#Mengubah tipe data date

df['starttime'] =  pd.to_datetime(df['starttime'], format="%m/%d/%Y %H:%M:%S")


# In[7]:


df['stoptime'] =  pd.to_datetime(df['stoptime'], format="%m/%d/%Y %H:%M:%S")


# In[8]:


#Null ada pada kolom birth year
#Namun pada modelling ini kolom birth year tidak digunakan dan akan dihapus
#Sehingga tidak dilakukan penggantian value

df.isnull().sum()


# #### Feature engineering: memproses data tanggal

# In[9]:


df['start_day'] = df.starttime.apply(lambda x: calendar.day_name[x.weekday()])


# In[10]:


df.sample(5)


# In[11]:


df['is_weekend'] = df.start_day.apply(lambda x: 1 if (x == 'Saturday' or x == 'Sunday') else 0)


# In[12]:


df.sample(5)


# In[13]:


def time_of_day(x):
    if x.hour < 6 or x.hour >= 22:    #### COMPLETE THE FUNCTION BELOW ####
        return 'night'
    elif x.hour > 18 and x.hour < 22:
        return 'evening'
    elif x.hour >= 12 and x.hour <= 18:
        return 'afternoon'
    else:
        return 'morning'


# In[14]:


df['start_moment'] = df.starttime.apply(time_of_day)
col = ['starttime', 'start_moment']
df[col].sample(5)


# #### Feature engineering: circle trip

# In[15]:


df['is_circle_trip'] = df.apply(lambda x: 1 if x['start station id'] == x['end station id'] else 0, axis = 1)
df.sample(3)


# #### Feature engineering: distance & trip duration

# In[16]:


from haversine import haversine


# In[17]:


def distance_stations(x):
    start_lat = x['start station latitude']
    start_long = x['start station longitude']
    end_lat = x['end station latitude']
    end_long = x['end station longitude']
    return haversine((start_lat,start_long),(end_lat,end_long))


# In[18]:


df['traveled_distance'] = df.apply(distance_stations, axis = 1)


# In[19]:


df.sample(3)


# In[20]:


df['average_speed'] = df.apply(lambda x: x['traveled_distance']/(x['tripduration']/3600), axis=1)


# In[21]:


col = ['traveled_distance', 'tripduration', 'average_speed']
df[col].sample(5)


# #### Variable encoding

# In[22]:


for variable_name in ['start_day','is_weekend',
                      'start_moment','is_circle_trip']:
    print('Dummifying the {} variable ...'.format(variable_name))
    dummies = pd.get_dummies(df[variable_name])
    dummies.columns = ['{}_{}'.format(variable_name,x) for x in dummies.columns]
    df = pd.concat([df,dummies],axis=1)


# In[23]:


dummy = ['start_day_Friday',
       'start_day_Monday', 'start_day_Saturday', 'start_day_Sunday',
       'start_day_Thursday', 'start_day_Tuesday', 'start_day_Wednesday',
       'is_weekend_0', 'is_weekend_1', 'start_moment_afternoon',
       'start_moment_evening', 'start_moment_morning', 'start_moment_night',
       'is_circle_trip_0', 'is_circle_trip_1']
df[dummy].sample(5)


# In[24]:


df.sample(5)


# In[25]:


for variable_name in ['start_day','is_weekend',
                      'start_moment','is_circle_trip']:
    print('Deleting the {} variable ...'.format(variable_name))
    del df[variable_name]


# In[26]:


del df['starttime'], df['stoptime'], df['start station name'], df['end station name']
del df['gender'], df['birth year']
del df['bikeid']
del df['start station id'], df['end station id']


# In[27]:


df.sample(5)


# #### Split training & testing

# In[28]:


labels = np.array(df.usertype)
features = np.array(df.drop(columns='usertype'))


# In[29]:


import sklearn
from sklearn.preprocessing import label_binarize
binarized_labels = label_binarize(labels, classes=['Customer', 'Subscriber']).ravel() 


# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, binarized_labels, test_size=0.3, random_state = 27)


# ### Modelling

# #### Logistic Regression

# In[31]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
lr = LogisticRegression()
score = cross_val_score(lr, X_train, y_train, scoring='roc_auc', cv=3)
print(score)
print('Logistic Regression Average Score: ', score.mean())


# #### KNN

# In[32]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
score = cross_val_score(knn, X_train, y_train, scoring='roc_auc', cv=3)
print(score)
print('KNN Average Score: ', score.mean())


# #### Naive Bayes

# In[33]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
score = cross_val_score(nb, X_train, y_train, scoring='roc_auc', cv=3)
print(score)
print('Naive Bayes Average Score: ', score.mean())


# #### Random Forest Classifier

# In[34]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10)
score = cross_val_score(rf, X_train, y_train, scoring='roc_auc', cv=3)
print(score)
print('Random Forest Average Score: ', score.mean())


# Nilai roc auc terbesar dimiliki oleh random forest classifier
