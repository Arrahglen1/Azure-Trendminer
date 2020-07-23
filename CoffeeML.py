#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from pandas import Series, DataFrame
from pylab import rcParams
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


df = pd.read_csv(r"C:\Users\arrah\Desktop\Trendminer_Intern\CoffeeML.csv", parse_dates=True)

count_row = df.shape[0]  # gives number of row count
count_column = df.shape[1] # gives number of column count

print('Number of rows: {}'.format(count_row))
print('Number of columns: {}'.format(count_column))

df.head()


# In[3]:


df.dtypes


# # Removoing duplicates values from the variable timestamp

# In[4]:


df1 = df.drop_duplicates(["timestamp"])


# In[5]:


count_row = df1.shape[0]  # gives number of row count
count_column = df1.shape[1] # gives number of column count
print('Number of rows: {}'.format(count_row))
print('Number of columns: {}'.format(count_column))


# In[6]:


df1.nunique()


# In[7]:


df1 = df1.set_index('timestamp')
df1.head(3)


# # checking if there is any missing values

# In[8]:


df1.isnull().values.any()


# # checking for the total number of missing values in each column

# In[9]:


df1.isnull().sum()


# # Descriptive statistics

# In[10]:


df1.describe()


# # corellation plot 

# In[11]:


plt.figure(figsize=(10,10))
p=sns.heatmap(df1.corr(), annot=True,cmap='RdYlGn',square=True, fmt='.2f')


# In[12]:


# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})


# In[13]:


df1['TemperatureAndHumiditySensor.temperature'].plot(linewidth=0.5);


# In[ ]:


cols_plot = ['TemperatureAndHumiditySensor.humidity', 'TemperatureAndHumiditySensor.temperature', 'AnalogNoiseSensor.noiseDiff ']
axes = df1[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('AnalogVibrationSensor.vibrationTotal')


# # Visualizing time series

# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 5, 4


# In[15]:


df1['TemperatureAndHumiditySensor.temperature'].plot()


# In[16]:


df2 = df1.sample(n=100, random_state=25, axis=0)

plt.xlabel('timestamp')
plt.ylabel('TemperatureAndHumiditySensor.humidity')
plt.title('Humidity*time plot')

df1['TemperatureAndHumiditySensor.temperature'].plot()


# In[ ]:




