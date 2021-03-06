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
#from matplotlib import rcParams

#from subsidary.config import cfg
plt.rc('text', usetex=True)

def define_initerval_time(sample_size_f,time_start_sec,t_downscale_f,params):
# Find start/end time (indices) for prediction, length of training interval, length of prediction interval
# for downsampled time series
# Input parameters: N_steps_f -; 
#                   sample_size_f -;  
#
#
#
#
#
#-----------------------------------------------------------------------------------------------
   ind_s = [i for i in range(len(t_downscale_f)) if t_downscale_f[i]  <= time_start_sec and t_downscale_f[i+1]  > time_start_sec]
   ind_start = ind_s[0]
   print(ind_start,sample_size_f)
   sS_f = [i for i in range(len(t_downscale_f)) if i == ind_start-sample_size_f] 
   ind_f_e = [i for i in range(len(t_downscale_f)) if i == ind_start+sample_size_f]
   ind_f_end=ind_f_e[0]
   ind_f_s = [i for i in range(len(t_downscale_f)) if i == ind_start]
   ind_f_start=ind_f_s[0]
   if ind_f_end<=ind_f_start:
    ind_f_end = ind_f_start+1
   sample_size_d=ind_f_start-sS_f[0]+1 # length of time series window (downsampled time series) used as training interval 
   window_size_d=ind_f_end-ind_f_start+1 # length of prediction interval 
   return ind_f_start,ind_f_end,sample_size_d,window_size_d

def read_data(params):
    f1 = open(params["file_data"]+".dat",'r')
    i=0
    for line in f1:
  #   content_array = np.loadtxt(f1, delimiter='\t', usecols=(0, 12), unpack=True)
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
def convert_to_secs(t):
# The input time should be given in format "dd/mm/yyyy"   
    t=t.rstrip()
    print('t.rstrip()',t[0:2],t[3:5],t[6:10],t[11],t[12],\
          t[14],t[15],t[17],t[18])
  #  t=t.split('/')
    ti_f=(int(str(t[6:10])),int(str(t[3:5])),int(str(t[0:2])),int(str(t[11])),int(str(t[12])),\
          int(str(t[14])),int(str(t[15])),int(str(t[17])),int(str(t[18])))
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

def save_data(t_in,X_in,file_write_data,params):
# Open new file and save data
 #   file_write_data = file_name + "_original.dat"
    fmt = joinStrings('{} ', np.size(X_in,0)+1)
    fmt='{}\n'
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

def plot_data(t_f,X_f,X_std,t_true,X_true,categories_f,categories_true,params):
  with open(params['file_summary_prediction']+".json", 'r') as f:
    data=json.load(f)
  f.close()
  params['session_id']=data['Session ID']
  t_data = data['Time start, [s]']
  t_date=convert_sec_date(int(t_data[:-2]))
  summary_text = 'Time start, [s] ='+ str(t_date) +' Parameter: ' +data['Parameter'] +' Units: '+ data['Units']+ '\n'\
                  +' Horizon, hours= '+str(data['Prediction horizon, hours']) +\
                 '  interp. points='+str(data['Number of points for interpolation '])+\
                 '  Nr inputs ='+str(data['Number of inputs'])+'  Nr outputs ='+str(data['Number of outputs'])
  labels2=['min','mean','max']
  labels1=['Pred','Observ']
  max_v=np.zeros(5)
  min_v=np.zeros(5)
  med_v=np.zeros(5)
  max_v[0]=float(data['Maximum value LSTM'])
  min_v[0]=float(data['Minimum value LSTM'])
  med_v[0]=float(data['Mean value LSTM'])
  max_v[1]=float(data['Maximum value LSTM EncodDecod'])
  min_v[1]=float(data['Minimum value LSTM EncodDecod'])
  med_v[1]=float(data['Mean value LSTM EncodDecod'])
  max_v[2]=float(data['Maximum value LSTM EncodDecod CNN'])
  min_v[2]=float(data['Minimum value LSTM EncodDecod CNN'])
  med_v[2]=float(data['Mean value LSTM EncodDecod CNN'])
  max_v[3]=float(data['Maximum value LSTM EncodDecod Conv'])
  min_v[3]=float(data['Minimum value LSTM EncodDecod Conv'])
  med_v[3]=float(data['Mean value LSTM EncodDecod Conv'])

  x=[1.0, 2.0]
  
  nVar_f=X_f.shape[0]
  ind_joint_categories = []
  for k in range(len(categories_f)): 
   for i in range(len(categories_true)):
     print(categories_true[i],i,categories_f[k],k) 
     if categories_true[i]==categories_f[k]:
        ind_joint_categories.append(i)
  s1 = 0
  ind_select=ind_joint_categories[0]
  for k in range(len(t_f)):
   for i in range(len(t_true)-1):
     if (t_true[i]<=t_f[k]) and (t_true[i+1]>t_f[k]):
       s1 = s1 + 1
  lengthT = s1
  rmse = np.zeros([lengthT,1]) 
  time = np.zeros(lengthT)
  X_original = np.zeros([lengthT,1]) 
  s1 = 0
  X_mean=np.float32(np.mean(X_f[:,0]))
  X_max = np.float32(max(X_f[:,0]))
  X_min = np.float32(min(X_f[:,0]))
  if lengthT>0: 
   for k in range(len(t_f)):
     for i in range(len(t_true)-1):
       if (t_true[i]<=t_f[k]) and (t_true[i+1]>t_f[k]):
       #  rmse[s1,0]=abs(X_f[k,0]-X_true[i,ind_joint_categories[0]]) # calculate RMSE for available prediction
       #  X_original[s1,0]=X_true[i,ind_joint_categories[0]] # create a copy of original time series
         for s2 in range(params['num_models']):
           rmse[s1,0]=abs(X_f[k,s2]-X_true[i,0]) # calculate RMSE for available prediction
         X_original[s1,0]=X_true[i,0] # create a copy of original time series 
         time[s1]=t_f[k] # create a copy of time series
         s1 = s1 + 1
  max_v[4]=np.max(X_original[:,0])
  min_v[4]=np.min(X_original[:,0])
  med_v[4]=np.mean(X_original[:,0])       
# Convert from seconds to datetime and to hours format ---------------------
  list_times_true = []
  list_times_f = []
  for i in range(len(t_f)):
   secs = int(t_f[i])  
   dt_object = datetime.fromtimestamp(secs)
   str_time = dt_object.strftime("%d-%b-%Y (%H:%M:%S)")
   list_times_f.append(str_time)
  # print("dt_object =", str_time)
  for i in range(len(t_true)):
   secs = int(t_true[i]) 
   dt_object = datetime.fromtimestamp(secs)
   str_time = dt_object.strftime("%d-%b-%Y (%H:%M:%S)")
   list_times_true.append(str_time)
  # print("dt_object =", str_time)   
# Plot results -------------------------------------------------
  fig = plt.figure()
 # fig, ax1 = plt.subplots(params['num_models'],3,sharex=True, sharey=True)
 # gs = fig.add_gridspec(1,params['num_models'], hspace=0)
  #ax1 = gs.subplots(sharex=True, sharey=True)
  fig.suptitle('Predicted versus observed data', fontsize=14, fontweight='bold')
  fig.subplots_adjust(left=0.15, bottom=None, right=None, top=0.85, wspace=None, hspace=0)
  models_collection =['LSTM','LSTM_EncodeDecode','LSTM_EncodeDecodeCNN','LSTM_EncodeDecodeConv']
  xticks = time[0:len(time):10]
  degrees=40
  date_labels=[]
  tt=params['time_start']
  t_date1=str(tt[0:10])+"\t\t"
  t_date2=str(tt[11:19])
  for k in range(len(time)):
     str_date = convert_sec_date(time[k])
     date_labels.append(str_date[0:22])
  for i1 in range(params['num_models']):
  #  ax1 = fig.add_subplot(4,1,i1+1)
    ax1=plt.subplot2grid((4,10),(i1,0),colspan=5)
    if lengthT>0:
     plt.plot(time,X_original[:,0],label='Original data')
    plt.plot(t_f,X_f[:,i1],label='Predicted data')
  #  textstr='Here is text'
    plt.gcf().text(0.2, 0.87, summary_text, fontsize=7)
    plt.fill_between(t_f, X_f[:,i1] - X_std[:,i1], X_f[:,i1] + X_std[:,i1], alpha=0.2)
 #   ax1[i1].plot(t_before_f,X_before_f,label='Original data, Predictor sequence')
 #  ax1.errorbar(t_f,X_f[:,0], yerr=X_std[:,ind_joint_categories[0]], label='both limits (default)')
  #  ax1[i1].errorbar(t_f,X_f[:,i1], yerr=X_std[:,i1],label='both limits (default)')
    if i1 == 0:
     plt.legend(('Observ.', 'Pred.'),
           loc='lower right',prop={'size': 6})
    plt.ylabel('Oxygen, ml/l', {'color': 'C0', 'fontsize': 10})
    if i1 == params['num_models'] -1:
     plt.xlabel('Time', {'color': 'C0', 'fontsize': 20})
    if i1 == params['num_models']-1:
     plt.xticks(time[0:len(time):10], date_labels[0:len(time):10], rotation=degrees)
    # plt.xlabel(date_labels[0:len(time):10])
    plt.tick_params(labelsize=6)
    max_all=np.max(max_v)
    min_all=np.min(min_v)
    marg=0.25*abs(max_all-min_all) 
    plt.ylim(min_all-marg,max_all+marg)
    models_collection[i1]=models_collection[i1].replace('_',' ')
    print(models_collection[i1],i1)
    plt.text(time[5], np.max(X_f[:,i1]), str(models_collection[i1]))
    ax2=plt.subplot2grid((4,10),(i1,6),colspan=2)
    y=np.array([med_v[i1], med_v[4]])
    yerr_min=np.array([min_v[i1], min_v[4]])#, [med_v[4]-min_v[4], max_v[4]-med_v[4]])
    yerr_max=np.array([max_v[i1], max_v[4]])#,
    bar_width=0.45
    plt.bar(x, abs(yerr_max-y),bar_width,color=['blue','pink'],bottom=y) #=yerr, fmt='.k', ecolor='gray', lw=1)
    plt.bar(x, abs(y-yerr_min), bar_width,color=['blue','pink'],bottom=yerr_min)
    if i1 == params['num_models']-1:
     plt.xticks(x, labels1)
    # plt.xlabel(labels1, fontsize=6, rotation = 45)
    plt.ylim(min_all-marg,max_all+marg)
    plt.hlines(med_v[i1], x[0]-0.5, x[0]+0.5, colors='k', linestyles='solid')
    plt.hlines(med_v[4], x[1]-0.5, x[1]+0.5, colors='k', linestyles='solid')
    plt.text(x[0]-0.5, min_v[i1]-marg/2, labels2[0], fontsize=6)
    plt.text(x[0]-0.5, med_v[i1]+marg/4, labels2[1], fontsize=6)
    plt.text(x[0]-0.5, max_v[i1]+marg/2, labels2[2], fontsize=6)
    plt.text(x[1]-0.5, min_v[4]-marg/2, labels2[0], fontsize=6)
    plt.text(x[1]-0.5, med_v[4]+marg/4, labels2[1], fontsize=6)
    plt.text(x[1]-0.5, max_v[4]+marg/2, labels2[2], fontsize=6)
    plt.xlim(0,3)     
  
  json_file = open(params['file_summary_prediction']+".json", 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  
 # ax2.plot(t_true,X_true[:,ind_joint_categories[0]],label='Full Time series')
 #  textstr = '\n'.join((
 #   r'Mean value= %5s' % (str(X_mean), ),
 #   r'Max value= %5s' % (str(X_max), ),
 #   r'Min value=%5s' % (str(X_min), ),
 #   r'Start time = %s %s' % (t_date1,t_date2, ),
 #   r'Horizon (hours)=%10s' % ( params['P_horizon_hours'])))
 # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
 #       verticalalignment='top')
 # ax2.legend()
  #fig.tight_layout()
  fig.show()
  plt.savefig(params["dir_prediction"]+"/"+params['PredictionPlot']+".pdf",bbox_inches='tight',dpi=100)

'''  fig = plt.figure()
  fig.suptitle('Predicted and original data', fontsize=14, fontweight='bold')
  ax1 = fig.add_subplot(111)
  fig.subplots_adjust(top=0.65)
#  rcParams['font.family'] = 'sans-serif'
  fig.suptitle('Predicted and original data')
  xticks = time[0:len(time):500]
  degrees=60
  date_labels=[]
  tt=params['time_start']
  t_date1=str(tt[0:10])+"\t\t"
  t_date2=str(tt[11:19])
  for k in range(len(t_true)):
    str_date = convert_sec_date(t_true[k])
    date_labels.append(str_date[0:22])
 # plt.xticks(t_true[0:len(t_true):500], date_labels[0:len(t_true):500], rotation=degrees)
  ax1.plot(t_true,X_true[:,ind_joint_categories[0]],label='Full Time series')
  plt.xticks(t_true[0:len(t_true):1000], date_labels[0:len(t_true):1000], rotation=degrees)
 # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
    #    verticalalignment='top')
  ax1.set_ylabel('Oxygen, ml/l', {'color': 'C0', 'fontsize': 10})
  ax1.set_xlabel('Time', {'color': 'C0', 'fontsize': 10})
  ax1.set(xlim=(t_true[0], t_true[len(t_true)-1]), \
          ylim=(np.min(X_true[:,ind_joint_categories[0]]), np.max(X_true[:,ind_joint_categories[0]])))
  fig.tight_layout()
  fig.show()
  plt.savefig(params["dir_prediction"]+"/"+"OriginalData.pdf",bbox_inches='tight',dpi=100) '''

def convert_sec_date(time_secs):
   dt_object = datetime.fromtimestamp(time_secs)
   str_time = dt_object.strftime("%d-%b-%Y (%H:%M:%S)")
   return str_time

def prepare_data(data_X_f,data_T_f,params):
#  Prepare data to standartized form according to selected training
#  interval
   len_data=np.size(data_X_f,1)
   n_Var_f=np.size(data_X_f,0)
   print('len_data',len_data,n_Var_f)
# Standartize data
   mu = np.zeros(n_Var_f,dtype=np.float32)
   sig = np.zeros(n_Var_f,dtype=np.float32)
   max_X_f = np.zeros(n_Var_f,dtype=np.float32)
   for si in range(n_Var_f): 
     mu[si] = np.mean(data_X_f[si][:])
     sig[si] = np.std(data_X_f[si][:])
     data_X_f[si][:]=(data_X_f[si][:]-mu[si])
     max_X_f[si]=max(np.abs(data_X_f[si][:]))
     data_X_f[si][:]=data_X_f[si][:]/max_X_f[si]
   t_true_f = np.zeros(len_data,dtype=np.float32)
   X_true_f = np.zeros((len_data,n_Var_f),dtype=np.float32)  
   t_true_f = data_T_f
   X_true_f = data_X_f
   P_horizon_sec_f=params['P_horizon_hours']*3600
   m_points=int(len_data/params['n_points']) # length of downsampled time series 
# Create downscaled time series 
   X_downscale_f=np.zeros([n_Var_f,m_points])
   t_downscale_f=np.zeros(m_points,dtype=np.float32)
   for si in range(n_Var_f):                           
     for i in range(m_points):
       if (i+1)*params['n_points']<=len_data:
         X_downscale_f[si][i]=np.mean(data_X_f[si][i*params['n_points']:(i+1)*params['n_points']])
         t_downscale_f[i]=data_T_f[i*params['n_points']]
         
# Calculate the number of time steps in the downsampled time series that correspond to prediction horizon     
   deltaT=t_downscale_f[1]-t_downscale_f[0]
   dT=t_true_f[1]-t_true_f[0]
   sample_size_f=int(P_horizon_sec_f/deltaT)
# Assign data for training
# Entire data for training
   length_train = m_points-1
   XTrain_f = np.zeros((n_Var_f,length_train),dtype=np.float32)
   YTrain_f = np.zeros((n_Var_f,length_train),dtype=np.float32)                          
   XTrain_f[si][:] = X_downscale_f[si][0:length_train]
   YTrain_f[si][:] = X_downscale_f[si][1:length_train+1]
   return t_true_f,X_true_f,t_downscale_f,X_downscale_f,XTrain_f,YTrain_f,sample_size_f,mu,sig,max_X_f

def save_network_json(model,model_name,params):
  model_json = model.to_json()
  with open(params["dir_network"]+"/"+model_name+".json","w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
  model.save_weights(params["dir_network"]+"/"+model_name+".h5")

def load_network_json(model_name,params):
  json_file = open(params["dir_network"]+"/"+model_name+".json", 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
# load weights into new model
  loaded_model.load_weights(params["dir_network"]+"/"+model_name+".h5")
  return loaded_model

def find_name_network(network_names_KF_Q,network_names_KF_R,network_names,shift_n,params):
  network_files_indir = []
  for (dirpath, dirnames, filenames) in walk(params["dir_network"]):
    network_files_indir.extend(filenames)
    break 
  network_exists= [0 for i in range(params['num_shifts'])]
  network_KF_Q_exists= [0 for i in range(params['num_shifts'])]
  network_KF_R_exists= [0 for i in range(params['num_shifts'])]
  for i in range(shift_n.shape[0]):
    for j in range(len(network_files_indir)):
      if network_names_KF_Q[i] + ".json" == network_files_indir[j]:
        network_KF_Q_exists[i]=1
      if network_names_KF_R[i] + ".json" == network_files_indir[j]:
        network_KF_R_exists[i]=1  
      if network_names[i] + ".json"== network_files_indir[j]:
        network_exists[i]=1
  return network_exists,network_KF_Q_exists,network_KF_R_exists

def set_network_name(shift_S,params):
  network_name = params['file_network']+"_"+str(shift_S)
  network_name_KF_Q = params['file_network_KF_Q']+"_"+str(shift_S)
  network_name_KF_R = params['file_network_KF_R']+"_"+str(shift_S)
  return network_name_KF_Q,network_name_KF_R,network_name

'''def run_model(params,network_exists,network_KF_exists,shift_n,...):
  for r1 in range(shift_n):
    if network_exists[r1]==0:
       train_model(params)
       if params['Algorithm_Scheme'] =="3LSTM_PATTERN_KF":
         tracker = multistep_lstm_kf(params)
       network_name_KF,network_name=set_network_name(shift_n[r1])
       save_network_json(network_name_KF)
       save_network_json(network_name)
    else:
       if params['Algorithm_Scheme'] =="3LSTM_PATTERN_KF":
         tracker = multistep_lstm_kf(params)'''
    
   
def save_prediction(n_var_f,t_stamp_prediction,X_prediction, X_std, \
                    t_stamp_history,X_history,params):
  dataset_prediction = np.zeros([t_stamp_prediction.shape[0],3])
  dataset_history = np.zeros([t_stamp_history.shape[0],1+n_var_f])
  for j in range(len(t_stamp_prediction)):
     dataset_prediction[j,0]=t_stamp_prediction[j]
     dataset_prediction[j,1]=X_prediction[j,0]
     dataset_prediction[j,2]=X_std[j,0]
  print('shape',X_history.shape,len(t_stamp_history))   
  for j in range(len(t_stamp_history)):
     dataset_history[j,0]=t_stamp_history[j]
     for i1 in range(n_var_f):
       dataset_history[j,1+i1]=X_history[j,i1]
  w = csv.writer(open(params["dir_prediction"]+"/"+params['file_output_history']+".csv", "w",newline=""))
  connect_history=[]
  for i1 in range(n_var_f):
     connect_history.append(dataset_history[j,i1])
  for j in range(t_stamp_history.shape[0]):
   connect_history=[dataset_history[j,0]]
   for i1 in range(n_var_f-1):
     connect_history.append(dataset_history[j,i1+1])
   w.writerow(connect_history)
  w = csv.writer(open(params["dir_prediction"]+"/"+params['file_output_prediction']+".csv", "w",newline=""))
  for j in range(t_stamp_prediction.shape[0]):
    w.writerow([dataset_prediction[j,0], dataset_prediction[j,1], dataset_prediction[j,2]])     
 # dataset_prediction.to_csv("dir_prediction"]+"/"+params["file_output_prediction"] +".csv")
 # dataset_history.to_csv("dir_prediction"]+"/"+params["file_output_history"] +".csv")
# Save mean , max and min predicted values of parameter in separate "csv" and "json" files

def save_summary_prediction(name_parameter,prediction_interval,t_stamp_prediction, \
                            data_prediction_mean, data_prediction_max, data_prediction_min,\
                            data_max, data_min, data_mean, params):

  
  dict = {'Time start, [s]' : str(t_stamp_prediction[0]), 'Parameter' : name_parameter, 'Units' : ' ml/l ', 'Prediction time' : prediction_interval, \
          'Maximum value LSTM' : str(data_prediction_max[0]), 'Minimum value LSTM' : str(data_prediction_min[0]), 'Mean value LSTM' : str(data_prediction_mean[0]),\
          'Maximum value LSTM EncodDecod' : str(data_prediction_max[1]), 'Minimum value LSTM EncodDecod' : str(data_prediction_min[1]), \
          'Mean value LSTM EncodDecod' : str(data_prediction_mean[1]),\
          'Maximum value LSTM EncodDecod CNN' : str(data_prediction_max[2]), 'Minimum value LSTM EncodDecod CNN' : str(data_prediction_min[2]), \
          'Mean value LSTM EncodDecod CNN' : str(data_prediction_mean[2]),\
          'Maximum value LSTM EncodDecod Conv' : str(data_prediction_max[3]), 'Minimum value LSTM EncodDecod Conv' : str(data_prediction_min[3]), \
          'Mean value LSTM EncodDecod Conv' : str(data_prediction_mean[3]),\
          'Maximum value Observ' : str(data_max), 'Minimum value Observ' : str(data_min), 'Mean value Observ' : str(data_mean),\
           'Prediction horizon, hours': str(params['P_horizon_hours']), 'Number of points for interpolation ': str(params['n_points']), \
          'Number of inputs': str(params['n_input']), 'Number of outputs': str(params['n_output']), 'Session ID': str(params['session_id'])}
  w = csv.writer(open(params['file_summary_prediction']+".csv", "w",newline=""))
  for key, val in dict.items():
    w.writerow([key, val])
  with open(params['file_summary_prediction']+".json","w") as f:
     json.dump(dict,f)
  f.close()  
    
def load_summary_from_csv(params):
    f = open(params['file_summary_prediction']+".csv",'r')
    csv_reader = csv.reader(f, delimiter=',')
    line_count = 0
    data ={}
    for row in csv_reader:
       data.update( {row[0] : row[1]} )
    f.close()    
    return data
   
def load_summary_from_json(params):
    f = open(params['file_summary_prediction']+".json",'r')
    data=f.read()
    f.close()    
    return eval(data)
 
def check_time_start(start_t, data_t):
  mess= ""
  if data_t[0] > start_t:
     mess = "Start time is too early, chose another time"
  elif data_t[-1] < start_t:
     mess = "Start time is too late, chose another time"
  return mess   
 
def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()

