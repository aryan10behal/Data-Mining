'''DO NOT DELETE ANY PART OF CODE
We will run only the evaluation function.

Do not put anything outside of the functions, it will take time in evaluation.
You will have to create another code file to run the necessary code.
'''

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE as tsne
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# other functions

def predict(test_set) :
    # find and load your best model
    # Do all preprocessings inside this function only.
    # predict on the test set provided
    # label_encoder object knows how to understand word labels.
  test_df = pd.read_csv(test_set) 
  data = pd.read_csv('covtype_train.csv')
  label_encoder = preprocessing.LabelEncoder()  

  data['Elevation']= label_encoder.fit_transform(data['Elevation']) 
  data['Aspect']= label_encoder.fit_transform(data['Aspect']) 
  data['Slope']= label_encoder.fit_transform(data['Slope'])
  data['Wilderness']= label_encoder.fit_transform(data['Wilderness'])
  data['Soil_Type']= label_encoder.fit_transform(data['Soil_Type']) 
  data['Hillshade_9am']= label_encoder.fit_transform(data['Hillshade_9am'])
  data['Hillshade_Noon']= label_encoder.fit_transform(data['Hillshade_Noon']) 
  data['Horizontal_Distance_To_Hydrology']= label_encoder.fit_transform(data['Horizontal_Distance_To_Hydrology']) 
  data['Vertical_Distance_To_Hydrology']= label_encoder.fit_transform(data['Vertical_Distance_To_Hydrology']) 
  data['Horizontal_Distance_To_Fire_Points']= label_encoder.fit_transform(data['Horizontal_Distance_To_Fire_Points']) 
  
  test_df['Elevation']= label_encoder.fit_transform(test_df['Elevation']) 
  test_df['Aspect']= label_encoder.fit_transform(test_df['Aspect']) 
  test_df['Slope']= label_encoder.fit_transform(test_df['Slope'])
  test_df['Wilderness']= label_encoder.fit_transform(test_df['Wilderness'])
  test_df['Soil_Type']= label_encoder.fit_transform(test_df['Soil_Type']) 
  test_df['Hillshade_9am']= label_encoder.fit_transform(test_df['Hillshade_9am'])
  test_df['Hillshade_Noon']= label_encoder.fit_transform(test_df['Hillshade_Noon']) 
  test_df['Horizontal_Distance_To_Hydrology']= label_encoder.fit_transform(test_df['Horizontal_Distance_To_Hydrology']) 
  test_df['Vertical_Distance_To_Hydrology']= label_encoder.fit_transform(test_df['Vertical_Distance_To_Hydrology']) 
  test_df['Horizontal_Distance_To_Fire_Points']= label_encoder.fit_transform(test_df['Horizontal_Distance_To_Fire_Points']) 
  
  for index,row in data.iterrows():
    row['target'] = row['target']-1

  features = data.loc[ : , data.columns != 'target']
  labels = pd.DataFrame(data['target'])
  test_df = test_df.loc[:, test_df.columns != 'target']

  kmeans = KMeans(n_clusters=5)

  kmeans.fit(features)
  y_pred = kmeans.predict(features)
  y_pred_list = list(y_pred)
  y_test_list = list(labels['target'])
  dic = {0:{0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 
        1:{0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 
            2:{0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 
              3:{0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 
                  4:{0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 
                    5:{0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 
                        6:{0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}}
  for i in range(len(y_pred_list)):
    index = y_pred_list[i]
    dic[index][y_test_list[i]] += 1
  mapping = {}
  for label in dic:
    new_dic = dic[label]
    max_key = max(new_dic, key=new_dic.get)
    mapping[label] = max_key
  # print(mapping)
  
  y_pred = kmeans.predict(test_df)

  for i in range(len(y_pred)):
    val = y_pred[i]
    y_pred[i] = mapping[val]
    y_pred[i] = y_pred[i]+1
  # f1 = metrics.f1_score(labels,y_pred, average='weighted')
  # print(f1)
    
  return list(y_pred)