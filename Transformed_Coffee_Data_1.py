#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from dateutil.parser import parse 
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import randn
from pandas import Series, DataFrame
from pylab import rcParams
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
###
from pandas import read_csv
from pandas import datetime


# In[ ]:


df = pd.read_csv(r"C:\Users\arrah\Desktop\Trendminer_Intern\CoffeeML1.csv", parse_dates =["timestamp"])

#df_large = read_csv(r"C:\Users\arrah\Desktop\Trendminer_Intern\CoffeeML1.csv",index_col=0,squeeze=True,parse_dates =["timestamp"])

count_row = df.shape[0]  # gives number of row count
count_column = df.shape[1] # gives number of column count

print('Number of rows: {}'.format(count_row))
print('Number of columns: {}'.format(count_column))

df.head()


# In[ ]:


#data_0 = df1.drop(df1['2020-07-16 00:00:00':'2020-07-16 07:50:00'].index)
df['timestamp'] = pd.to_datetime(df['timestamp'])
startDate=['2020-07-16 05:00:00','2020-07-17 05:00:00','2020-07-18 05:00:00','2020-07-19 05:00:00','2020-07-20 05:00:00','2020-07-21 05:00:00','2020-07-22 05:00:00']
endDate=['2020-07-16 17:00:00','2020-07-17 17:00:00','2020-07-18 17:00:00','2020-07-19 17:00:00','2020-07-20 17:00:00','2020-07-21 17:00:00','2020-07-22 17:00:00']
#data_0[data_0.timestamp<endDate & startDate]
#df[df['timestamp']>= startDate & df ['timestamp']<= endDate]
result = []
for i in  range(0, len(startDate)) :
    start = startDate[i]
    end = endDate[i]
    result.append((df['timestamp'] > start) & (df['timestamp'] <= end)) 

resultDf = []
for item in result:
    if len(item)>0:
        resultDf.append(df.loc[item])

merged_df =pd.concat(resultDf)
merged_df


# In[ ]:


df2= merged_df.set_index('timestamp')
df2


# In[ ]:


df2.index


# In[ ]:


df3=df2.resample('100ms',level=0).mean()
df3


# In[ ]:


X, y = df3.drop(['Keypad.keyPressed'], axis=1), df3['Keypad.keyPressed']
X


# In[ ]:


X_inter = X.interpolate(method='linear', axis=0).ffill().bfill()
X_inter


# In[ ]:


merged_df1 = pd.concat([X_inter, y], axis=1)
merged_df1


# In[ ]:


df_2 = merged_df1[['AnalogNoiseSensor.noiseTotal','Keypad.keyPressed']]
df_2


# In[ ]:


df_3= df_2.reset_index()
df_3


# In[ ]:


df_10 =df_3[['AnalogNoiseSensor.noiseTotal']]
df_10


# # Sliding Window

# In[ ]:


global_len=300000
global_k=60
#global_start=200000
# Applying only for the variable analognoise.total

def generate_sample(k, sample_list):
    result = []
    for i in range(len(sample_list)):
        if i < len(sample_list) - k:
            result.append(sample_list[i : i+k])           
    temp = np.array(result)
    final_result = np.reshape(temp, (1, len(sample_list)-k, k))
    return final_result
    


data_matrix = df_10[:global_len].values

sampled_matrix_of_k = generate_sample(global_k, data_matrix)

new_analog= pd.DataFrame(sampled_matrix_of_k[0])
new_analog
#sampled_matrix_of_k[0][0]


# In[ ]:


import math

def sliding_window_of_values(key_pad_frame, k):
    keypad_matrix = key_pad_frame.values
    
    r = []
    for i in range(len(keypad_matrix) - k):
        print(keypad_matrix[i : i+k])
        for y in range(len(keypad_matrix[i : i+k])):
            if not math.isnan(keypad_matrix[i : i+k][y]):
            
                r.append(keypad_matrix[i : i+k][y])
                break
            elif y == len(keypad_matrix[i : i+k]) - 1:
                r.append(keypad_matrix[i : i+k][y])
        
    return r
                
test_dataframe = df_3["Keypad.keyPressed"][:global_len]

keypadd = sliding_window_of_values(test_dataframe, global_k)


# In[ ]:


keypadd


# In[ ]:


print(len(keypadd))


# In[ ]:


df_timestamp =df_3[['timestamp']][:global_len-global_k]
df_timestamp


# # # concantenating dataframe

# In[ ]:


new_analog['keypad'] = keypadd


# In[ ]:


new_analog


# In[ ]:


merged_frame = pd.concat([df_timestamp, new_analog], axis=1)
merged_frame


# In[ ]:


# count the number of occurences greater than target_value in a list of integer
def count_occurence_in_row(row, target_value):
    count = 0
    for i in range(len(row)):
        if row[i] > target_value:
            count += 1
    return count


# In[ ]:


# returns true if the count in a row is greater or equal than the cutt_off_pt
def is_valid_row(row, target_value, cut_off_pt):
    if count_occurence_in_row(row, target_value) >= cut_off_pt:
        return True
    else:
        return False


# In[ ]:



def create_data_frame_with_valid_row(sample_data_frame, cutt_off_pt, target_value):
    analog_df_values = sample_data_frame.drop(columns=['timestamp', 'keypad']).values
    invalid_indexes = []
    for index in range(len(analog_df_values)):
        if not is_valid_row(analog_df_values[index], target_value, cutt_off_pt):
            invalid_indexes.append(index)
    return sample_data_frame.drop(sample_data_frame.index[invalid_indexes])


# In[ ]:


exemple_frame = merged_frame[:100]
exemple_frame.info()


# In[ ]:


mat = exemple_frame.drop(columns=['timestamp', 'keypad']).values

print(is_valid_row(mat[59], 15000, 40))

print(count_occurence_in_row(row=mat[59], target_value=15000))


# In[ ]:


result_frame = create_data_frame_with_valid_row(merged_frame, cutt_off_pt=40, target_value=15000)

result_frame


# In[ ]:


#using the index as a list to do the filtering
result_frame.index.tolist()


# In[ ]:


# the function uses the list and takes the first of consecutive sequence base on the difference we chose
def drop_consecutive_row(input_df, delta ):
    index_list = input_df.index.tolist()
    index_to_keep = [index_list[0]]
    for i in range(len(index_list)-1):
        if index_list[i+1] - index_list[i] > delta:
            index_to_keep.append(index_list[i+1])
    return input_df.loc[index_to_keep]


# In[ ]:


selecte_frame = drop_consecutive_row(result_frame, 50)
selecte_frame


# In[ ]:


#g = selecte_frame.to_csv(r"C:\Users\arrah\Desktop\Trendminer_Intern\selecte_frame_15000_50_1.csv", index = True)


# In[ ]:


X_indep , Y_dep= selecte_frame.drop(['timestamp', 'keypad'], axis=1), selecte_frame['keypad']
X_indep


# In[ ]:


#g = X1.to_csv(r"C:\Users\arrah\Desktop\Trendminer_Intern\x1__40.csv", index = True)


# In[ ]:


df_final = pd.read_csv(r"C:\Users\arrah\Desktop\Trendminer_Intern\drop_consecutive_row_15000_40.csv")
df_final


# In[ ]:


X_final , Y_final= df_final.drop(['timestamp', 'keypad'], axis=1), df_final['keypad']
X_final


# # Clustering Models
# setting up for cluster Analysis

# In[ ]:


import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(7,4))


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
#scaler = StandardScaler()

# Optionnal
X_scale= pd.DataFrame(scaler.fit_transform(X_final), columns=X_final.columns)
X_scale


# # Building and running your model

# In[ ]:


clusters = 5
kmeans = KMeans(n_clusters = clusters, random_state=123) 
y_assign=kmeans.fit(X_scale) 

print(y_assign.labels_)


# In[ ]:


type(y_assign)


# In[ ]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=123)
kmeans.fit(X_scale)
y_kmeans = kmeans.predict(X_scale)
y_kmeans


# In[ ]:


plt.scatter(X_scale[:, 0], X_scale[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# In[ ]:





# # Plotting your model outputs

# In[ ]:


X_scale.columns = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59']
Y_dep.columns = ['keypad']


# In[ ]:


#converting numpy to pandas
dr = pd.DataFrame(data=y_assign.flatten())
dr


# In[ ]:


# color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue'])

# plt.subplot(1,2,1)

# plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[iris.target], s=50)
# plt.title('Ground Truth Classification')

# plt.subplot(1,2,2)

# plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[clustering.labels_], s=50)
# plt.title('K-Means Classification')


# # Evaluate your clustering results

# In[ ]:


#relabel = np.choose(kmeans.labels_, [2, 0, 1, 3, 4]).astype(np.int64)


# In[ ]:


#print(classification_report(y_assign, relabel))


# In[ ]:


# clusters = 5
  
# kmeans = KMeans(n_clusters = clusters) 
# y =kmeans.fit_predict(result_frame2) 
# y 
# #print(y.labels_)


# In[ ]:


#converting numpy to pandas
# dr = pd.DataFrame(data=y.flatten())
# dr


# In[ ]:


#g = y.to_csv(r"C:\Users\arrah\Desktop\Trendminer_Intern\x1_15000_assign_40.csv", index = True)


# In[ ]:


result_frame2['cluster']= y
result_frame2.head


# In[ ]:


from sklearn.decomposition import PCA   
pca = PCA(3) 
pca.fit(result_frame2) 
  
pca_data = pd.DataFrame(pca.transform(result_frame2)) 
  
print(pca_data.head())


# # Hierachical Clustering

# In[ ]:


from scipy.cluster.hierarchy import dendrogram, linkage
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.metrics import silhouette_score 
import scipy.cluster.hierarchy as shc 


# In[ ]:


sts = X1.iloc[:,:].values
sts[:,0]


# # CLUSTERING Create Linkage Matrix

# In[ ]:


Z = linkage(sts, 'ward')


# Plot Dendrogram of Clusters

# In[ ]:


plt.figure(figsize = (25, 10))
plt.title('Hierachical Cluster Dendrogram ')
plt.axhline(y=90000, color='r', linestyle='--')
#plt.ylabel('distance')
dendrogram(
    Z,
    labels = X1.index,
    leaf_rotation = 0.,
    leaf_font_size = 18.,
)
plt.show()


# In[ ]:


from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(sts)


# In[ ]:


plt.figure(figsize=(10, 7))
plt.scatter(sts[:,0], sts[:,0], c=cluster.labels_, cmap='rainbow')


# In[ ]:


result_frame1=result_frame[:50]
result_frame1


# In[ ]:


result_frame1.index.tolist()


# In[ ]:


def drop_consecutive_row(input_df, delta ):
    index_list = input_df.index.tolist()
    index_to_keep = [index_list[0]]
    for i in range(len(index_list)-1):
        if index_list[i+1] - index_list[i] > delta:
            index_to_keep.append(index_list[i+1])
    return input_df.loc[index_to_keep]
        


# In[ ]:


drop_consecutive_row(result_frame1, 5)

