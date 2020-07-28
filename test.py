import os
#import utils
import csv
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
#from subsidary.config import cfg
plt.rc('text', usetex=True)

def define_initerval_time(t_f,time_sec_f):
# Find start/end time (indices) for prediction, length of training interval, length of prediction interval
# for downsampled time series      
   [sS_f for i, value in time_sec_f if value == time_sec_f[ind_start-sampleSize]] 
   [ind_f_end for i, value in time_sec_f if value == time_sec_f[ind_start+Nsteps]] 
   [ind_f_start for i, value in time_sec_f if value == time_sec_f[ind_start]] 
   if ind_f_end<=ind_f_start:
    ind_f_end = ind_f_start+1
   sample_size_f=ind_f_start-sS_f+1 # length of time series window (downsampled time series) used as training interval 
   window_size_f=ind_f_end-ind_f_start+1 # length of prediction interval 
   return init_time, sample_size_f, window_size_f

##def read_data():
##    points = np.arange(0, 12, 1)
##    cols = np.arange(0, 12, 1)
##    f1 = open('NormalizedBoknis.dat','r')
##    i=0
##    for line in f1:
##  #   content_array = np.loadtxt(f1, delimiter='\t', usecols=(0, 12), unpack=True)
##      line_array = line
##      sp_array=line_array.split('\t')
##      i=i+1
##    numrows=i+1  
##    f1.seek(0, 0)
##    content_array=[[0 for x in range(numrows)] for y in range(len(sp_array))]
##    i=0
##    for line in f1:
##      line_array = line
##      sp_array=line_array.split('\t')
##      for j in range(len(sp_array)):
##        content_array[j][i]=float(sp_array[j])
##      i=i+1
##    f1.close()
### Read list of initial categories
##    categories_init = list()
##    with open('InitialCategories.dat','r') as f2:
##      for line in f2: 
##        [categ] = line.split()
##        categories_init.append(categ) 
##    f2.close()        
### Read list of input categories - parameters
##    categories_def = list()
##    key_cat = list()
##    with open('ListCategories.dat','r') as f3:
##      for line in f3: 
##        [value_c, key_c]=line.split()
##        if int(value_c) == 1: key_cat.append(key_c)
##    f3.close()    
##    categories_def = key_cat
### Select data column according to input categories
##    categories_f=[]
##    ind=[]
##    X_data = [[0 for x in range(numrows)] for y in range(len(categories_def))]
##    t_data = [0 for x in range(numrows)] 
##    categories_f =set(categories_def).intersection(set(categories_init))
##    for i in range(len(categories_init)):
##        for j in range(len(categories_def)):
##            if categories_init[i] == categories_def[j]:
##                X_data[j][:]=content_array[i+1][:]
##                ind.append(i)
##    t_data =  content_array[0][:]          
##    return categories_f, X_data, t_data
def convert_to_secs(t):
# The input time should be given in format "dd/mm/yyyy"   
    t=t.split('/')
    ti_f=(int(t[2]), int(t[1]), int(t[0]), 0, 0, 0, 0, 0, 0)
    secs = time.mktime( ti_f )
    return secs
   
def load_data():
    f1 = open('Filter.dat','r')
    i=0
    first_line = f1.readline()
    categories_filter=first_line.split('\t')
    for i in range(len(categories_filter)):
       categories_filter[i]=categories_filter[i].rstrip()
    i=0   
    for line in f1:
  #   content_array = np.loadtxt(f1, delimiter='\t', usecols=(0, 12), unpack=True)
       line_array = line
       sp_array=line_array.split('\t')
       i=i+1
    numrows=i  
    content_array=[[0 for x in range(numrows)] for y in range(len(sp_array))]
    i=0
    f1.seek(0,0)
    for line in f1:
      line_array = line
      sp_array=line_array.split('\t')
      if i > 0:
       for j in range(len(sp_array)):
         content_array[j][i-1]=float(sp_array[j])
      i=i+1
    f1.close()
    t_filter=[0 for x in range(numrows)]
    X_filter = [[0 for x in range(numrows)] for y in range(len(sp_array)-1)]
    t_filter[0:numrows]=content_array[0][0:numrows]
    X_filter=content_array[1:len(sp_array)-1][0:numrows]
    f2 = open('True.dat','r')
    first_line = f2.readline()
    categories_true=first_line.split('\t')
    for i in range(len(categories_true)):
       categories_true[i]=categories_true[i].rstrip()
    i=0
    for line in f2:
  #   content_array = np.loadtxt(f1, delimiter='\t', usecols=(0, 12), unpack=True)
       line_array = line
       sp_array=line_array.split('\t')
       i=i+1
    numrows=i  
    content_array=[[0 for x in range(numrows)] for y in range(len(sp_array))]
    i=0
    f2.seek(0,0)
    for line in f2:  
       line_array = line
       sp_array=line_array.split('\t')
       if i > 0:
        for j in range(len(sp_array)):
         content_array[j][i-1]=float(sp_array[j])
       i=i+1
    f2.close()
    t_true=[0 for x in range(numrows)] 
    t_true[0:numrows]=content_array[0][0:numrows]
    X_true = [[0 for x in range(numrows)] for y in range(len(sp_array)-1)]
    X_true=content_array[1:len(sp_array)-1][0:numrows]
    f3 = open('Pred.dat','r')
    first_line = f3.readline()
    categories_f= first_line.split('\t')
    for i in range(len(categories_f)):
       categories_f[i]=categories_f[i].rstrip()
    i=0   
    for line in f3:
  #   content_array = np.loadtxt(f1, delimiter='\t', usecols=(0, 12), unpack=True)
      line_array = line
      sp_array=line_array.split('\t')
      i=i+1
    numrows=i  
    content_array=[[0 for x in range(numrows)] for y in range(len(sp_array))]
    i=0
    f3.seek(0,0)
    for line in f3:
       line_array = line
       sp_array=line_array.split('\t')
       if i > 0:
        for j in range(len(sp_array)):
         content_array[j][i-1]=float(sp_array[j])
       i=i+1
    t_f=[0 for x in range(numrows)]
    t_f[0:numrows]=content_array[0][0:numrows]
    X_f_mean = [[0 for x in range(numrows)] for y in range(len(sp_array)-1)]
    X_f_mean=content_array[1:len(sp_array)][0:numrows]
    f3.close()
    return t_f,X_f_mean,t_true,X_true,t_filter,X_filter, len(sp_array)-1,categories_true,categories_filter,categories_f

def save_data(t_in,X_in,file_name):
# Open new file and save data
##    file_predicted_data = file_name + "_pred.dat"
##    fmt = joinStrings('{} ', np.size(X_f_mean,0)+1)
##    fmt=fmt+'\n'
##    print(fmt)
##    content_array=[[0 for x in range(len(t_f))] for y in range(np.size(X_f_mean,0)+1)]
##    content_array[0][:]=t_f
##    for i in range(np.size(X_f_mean,0)):
##      content_array[i+1][:]=X_f_mean[i][:] 
##    with open(file_predicted_data,'w') as f1:
##       for j in range(np.size(content_array,1)):
##          tuple_cont=( )
##          for i in range(np.size(X_f_mean,0)+1):
##            tuple_cont = tuple_cont + (content_array[i][j],)
##          f1.write(fmt.format(tuple_cont[0],tuple_cont[1]))
##    f1.close()
##    file_original_data = file_name + "_original.dat"
##    fmt = joinStrings('{} ', np.size(X_true,0)+1)
##    fmt=fmt+'\n'
##    print(fmt)
##    content_array=[[0 for x in range(len(t_true))] for y in range(np.size(X_true,0)+1)]
##    content_array[0][:]=t_true
##    for i in range(np.size(X_true,0)):
##      content_array[i+1][:]=X_true[i][:] 
##    with open(file_original_data,'w') as f2:
##       for j in range(np.size(content_array,1)):
##          tuple_cont=( )
##          for i in range(np.size(X_true,0)+1):
##            tuple_cont = tuple_cont + (content_array[i][j],)
##          #  print(np.asarray(tuple_cont))
##          f2.write(fmt.format(tuple_cont[0],tuple_cont[1], tuple_cont[2]))
##    f2.close()
    file_write_data = file_name + "_original.dat"
    fmt = joinStrings('{} ', np.size(X_in,0)+1)
    fmt='{}\n'
    print(fmt)
    content_array=[[0 for x in range(len(t_in))] for y in range(np.size(X_in,0)+1)]
    content_array[0][:]=t_in
    for i in range(np.size(X_in,0)):
      content_array[i+1][:]=X_in[i][:] 
    with open(file_write_data,'w') as f1:
       for j in range(np.size(content_array,1)):
          tuple_cont=( )
          for i in range(np.size(X_in,0)+1):
            tuple_cont = tuple_cont + (str(content_array[i][j]),)
          str_t = ' '.join(tuple_cont) 
          f1.write(fmt.format(str_t))
    f1.close()

def joinStrings(string_e,num_str):
    list=""
    for e in range(num_str):
        list = list + string_e
    return list

##t_f,X_f_mean,t_true,X_true,t_filter,X_filter,numResponses=load_data()
##fileName="PRED"
##print(len(X_f_mean))
##save_data(t_true,X_true,fileName)

def plot_data(t_f,X_f,t_true,X_true,categories_f,categories_true):
  nVar_f=np.size(X_f,0)
  ind_joint_categories = []
  for k in range(nVar_f+1):
   for i in range(len(categories_true)):
     if str(categories_true[i])==str(categories_f[k]):
        ind_joint_categories.append(i)
  s1 = 0
  ind_select=ind_joint_categories[1]-1
  for k in range(len(t_f)):
   for i in range(len(t_true)):
     if (t_true[i]==t_f[k]):
       s1 = s1 + 1
  lengthT = s1
  rmse = [[0 for x in range(lengthT)] for y in range(1)] 
  time = [0 for x in range(lengthT)]
  X_original = [[0 for x in range(lengthT)] for y in range(1)] 
  s1 = 0
  print(ind_select)
  for k in range(len(t_f)):
   for i in range(len(t_true)):
     if (t_true[i]==t_f[k]):  
       rmse[0][s1]=abs(X_f[ind_select][k]-X_true[ind_select][i]) # calculate RMSE for available prediction
       X_original[0][s1]=X_true[ind_select][i] # create a copy of original time series 
       time[s1]=t_f[k] # create a copy of time series
       s1 = s1 + 1
      # print(t_true[i],t_f[k])
# Convert from seconds to datetime and to hours format ---------------------
  list_times_true = []
  list_times_f = []
  for i in range(len(t_f)):
   secs = int(t_f[i])
  for i in range(len(t_f)):
   secs = int(t_f[i])  
   dt_object = datetime.fromtimestamp(secs)
   str_time = dt_object.strftime("%d-%b-%Y (00:00:00)")
   list_times_f.append(str_time)
  # print("dt_object =", str_time)
  for i in range(len(t_true)):
   secs = int(t_true[i])
  print(secs) 
  for i in range(len(t_true)):
   secs = int(t_true[i]) 
   dt_object = datetime.fromtimestamp(secs)
   str_time = dt_object.strftime("%d-%b-%Y (00:00:00)")
   list_times_true.append(str_time)
  # print("dt_object =", str_time)   
# Plot results -------------------------------------------------
  fig, (ax1, ax2) = plt.subplots(2, sharex=True)
  fig.suptitle('Predicted and original data')
  ax1.plot(time[:],X_original[ind_select][:],t_f[:],X_f[ind_select][:],label='Original data')
  ax1.legend(('Original data', 'Predicted data'),
           loc='upper right')
  ax1.set_ylabel(r'\bf{Normalized Oxygen}', {'color': 'C0', 'fontsize': 20})
  ax1.set_xlabel(r'\bf{Time}', {'color': 'C0', 'fontsize': 20})
  ax2.plot(t_f[:],X_f[ind_select][:],label='Predicted data')
  ax2.legend()
  fig.show()
  
t_f,X_f_mean,t_true,X_true,t_filter,X_filter, numResponses,categories_true,categories_filter,categories_f = load_data()
print(t_f[0],t_f[len(t_f)-1])
plot_data(t_f,X_f_mean,t_true,X_true,categories_f,categories_true)

##def prepare_data(n_points_f,P_horizon_sec_f,define_choice_training_f,TrainInter_f, \
##                 n_Var_f,data_X_f,data_T_f):
###  Prepare data to standartized form according to selected training
###  interval
##   len_data=len(data_X_f[:][1])
##   X_downscale_f=np.zeros((float(len_data/2),n_Var_f),dtype=np.float32)
##   for si in range(n_Var_f):
### Standartize data
##     mu[si] = np.mean(data_X_f[:][si])
##     sig[si] = np.std(data_X_f[:][si])
##     data_X_f[:][si]=(data_X_f[:][si]-mu[si])/sig[si]
##     max_X_f=max(data_X_f[:][si])
##     data_X_f[:][si]=data_X_f[:][si]/max_X_f
### Create copy of original time series 
##     X_true_f[1][si]=data_X_f[1][si]
##     t_true_f[1]=data_T_f[1]
### Create downscaled time series 
##     X_downscale_f[1][si]=data_X_f[1][si]
##     t_downscale_f[1]=data_T_f[1]
##     num_points=float(len_data)
##     for i in range(num_points):
##       X_true_f[i][si]=data_X_f[i][si]
##       t_true_f[i]=data_T_f[i]
##     deltaT=t_true_f[2]-t_true_f[1]
##     len_d=len(X_true_f[:][si])
##     Nsteps_f=float(P_horizon_sec_f/deltaT)
##     m_points=float(len_data/n_points_f)
##     for i in range(1, m_points):
##       if i*n_points_f<=len_data:
##         X_d_f[i][si]=np.mean(data_X_f[1+(i-1)*n_points_f:i*n_points_f][si])
##         t_d_f[i]=data_T_f[1+(i-1)*n_points_f]
##     t_downscale_f=np.linspace(data_T_f[1],t_d_f[len(t_d_f)],float(len_data/2))
##     X_downscale_f[:][si]=interp1(t_d_f,X_d_f[:][si],t_downscale_f);
### Assign data for training
### Entire data for training
##     if define_choice_training_f == "FULL" and TrainInter_f==1: 
##       XTrain_f[:][si] = X_downscale_f[1:end-1][si]
##       YTrain_f[:][si] = X_downscale_f[2:end][si]
### Interval is used for training
##     elif define_choice_training_f == "PART":
##       XTrain_f[:][si] = X_downscale_f[1:TrainInter_f-1][si]
##       YTrain_f[:][si) = X_downscale_f[2:TrainInter_f][si]
##   return t_true_f,X_true_f,t_downscale_f,X_downscale_f,XTrain_f,YTrain_f,Nsteps_f

    
