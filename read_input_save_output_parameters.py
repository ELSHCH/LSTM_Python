import sys, os
#import utils
from os import walk
import csv
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from keras.models import model_from_json
import time
import json

def save_prediction(name_parameter,prediction_interval,t_stamp_prediction,X_prediction, \
                    t_stamp_history,X_history,params):
'''-----------------------------------------------------------------------------------
   Save prediction time series in .csv file and summary of prediction into .json file
   Variables: name_parameter - name of predicted parameter (in our case oxygen)
              prediction_interval - length of prediction horizon ( 12- 48 hours)
              t_stamp_prediction[] - array with time in seconds corresponding to prediction time
              X_prediction[] - array with predicted parameter data
              t_stamp_history[] - array with time stamps preceeding prediction time interval
              X_history[] - array with parameters used for prediction ( testing data) 
              params - global parameters defined by user see config.py
-------------------------------------------------------------------------------------'''
  dataset_prediction = np.zeros([t_stamp_prediction.shape[0],2])
  dataset_history = np.arange(2*t_stamp_history.shape[0]).reshape(t_stamp_history.shape[0],2)
  
  for j in range(t_stamp_prediction.shape[0]):
     dataset_prediction[j,0]=t_stamp_prediction[j]
     dataset_prediction[j,1]=X_prediction[j]   
  for j in range(t_stamp_history.shape[0]):
     dataset_history[j][0]=t_stamp_history[j]
     dataset_history[j][1]=X_history[j,0]
  w = csv.writer(open(params["dir_prediction"]+"/"+params['file_output_history']+".csv", "w",newline=""))
  for j in range(t_stamp_history.shape[0]):
    w.writerow([dataset_history[j,0], dataset_history[j,1]])
  w = csv.writer(open(params["dir_prediction"]+"/"+params['file_output_prediction']+".csv", "w",newline=""))
  for j in range(t_stamp_prediction.shape[0]):
    w.writerow([dataset_prediction[j,0], dataset_prediction[j,1]])     
 # dataset_prediction.to_csv("dir_prediction"]+"/"+params["file_output_prediction"] +".csv")
 # dataset_history.to_csv("dir_prediction"]+"/"+params["file_output_history"] +".csv")
# Save mean , max and min predicted values of parameter in separate "csv" and "json" files
  data_prediction_mean = np.mean(dataset_prediction[:][1],axis=0)
  data_prediction_max = max(dataset_prediction[:][1])
  data_prediction_min = min(dataset_prediction[:][1])
  dict = {'Time start, [s]' : str(t_stamp_prediction[0]), 'Parameter' : name_parameter, 'Units' : ' ml/l ', 'Prediction time' : prediction_interval, \
          'Maximum value' : str(data_prediction_max), 'Minimum value' : str(data_prediction_min), 'Mean value' : str(data_prediction_mean)}
  w = csv.writer(open(params["dir_prediction"]+"/"+params['file_summary_prediction']+".csv", "w",newline=""))
  for key, val in dict.items():
    w.writerow([key, val])
  with open(params["dir_prediction"]+"/"+params['file_summary_prediction']+".json","w") as f:
     json.dump(dict,f)
  f.close()

  def read_data(params):
'''-----------------------------------------------------------------------------------
   Read testing data and time stamps in seconds from text file 
   Variables: params - global parameters defined by user see config.py
-------------------------------------------------------------------------------------'''      
# Read ocean and atmospheric parameters from file and save data into content_array defined as list
    f1 = open(params["file_data"]+".dat",'r')
    i=0
    for line in f1:
      line_array = line
      sp_array=line_array.split('\t')
      i=i+1
    numrows=i  
    f1.seek(0, 0)
    content_array=[[0 for x in range(numrows)] for y in range(len(sp_array))]
    i=0
    for line in f1:
      line_array = line
      sp_array=line_array.split('\t')
      for j in range(len(sp_array)):
        content_array[j][i]=float(sp_array[j])
      i=i+1
    f1.close()
    
# Read list of initial categories
    categories_init = list()
    with open('InitialCategories.dat','r') as f2:
      for line in f2: 
        [categ] = line.split()
        categories_init.append(categ) 
    f2.close()        
# Read list of input categories - parameters
    categories_def_in = list()
    categories_def_out = list()
    key_cat_in = list()
    key_cat_out =list()
    with open('ListCategories.dat','r') as f3:
      for line in f3: 
        [value_in, value_out, key_c]=line.split()
        if int(value_in) == 1: key_cat_in.append(key_c)
        if int(value_out) == 1: key_cat_out.append(key_c)
    f3.close()    
    categories_def_in = key_cat_in
    categories_def_out = key_cat_out
# Select data column according to input categories
    categories_in=[]
    categories_out=[]
    ind=[]
    k = 0
    t_data = [0 for x in range(numrows)] 
    categories_in =set(categories_def_in).intersection(set(categories_init))
    X_data = [[0 for x in range(numrows)] for y in range(len(categories_in))]
    categories_out =set(categories_def_out).intersection(set(categories_init))
    for i in range(len(categories_init)):
        for j in range(len(categories_def_in)):
            if categories_init[i] == categories_def_in[j]:
                X_data[k][:]=content_array[i+1][:]
                k = k + 1
    list_categories_in= list(categories_in)
    list_categories_out= list(categories_in)
    for i in range(len(categories_in)):
        for j in range(len(categories_out)):
            if list_categories_in[i] == list_categories_out[j]:            
                ind.append(i)
    t_data =  content_array[0][:]          
    return  X_data, t_data, categories_in,categories_out,ind
