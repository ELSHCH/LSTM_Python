import os
#import utils
import csv
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from subsidary import dt_utils as dt_ut
import time

def sample_covariance(X_var_f,n_var_f,n_observ_f):
#------------------------------------------------------------------------
#    Estimate covariance matrix
#    Input parameters: X_var_f - time series of size (n_observ_f X n_var_f);
#                      n_var_f - number of parameters in time
#                      series;
#                      n_observ_f - number of time steps;
#    Output parameters: Q_covar - matrix of covariance of size (n_var_f X n_var_f)
#-------------------------------------------------------------------------
  Q_covar=np.zeros((n_var_f,n_var_f),dtype=np.float32)
  mean_X_var=np.zeros(n_var_f,dtype=np.float32)
  for k in range(n_var_f):    
      mean_X_var[k] = np.mean(X_var_f[:,k])  
      for i in range(n_var_f):    
        for j in range(n_observ_f):
          Q_covar[i,i]=Q_covar[i,i]+(X_var_f[j,i]- mean_X_var[i])**2   
      Q_covar[i,i]=Q_covar[i,i]/n_observ_f
  return Q_covar

def covariance_state_transition(X_var_f,X_Pred_f,n_var_f,n_observ_f):
#------------------------------------------------------------------------
#    Estimate covariance of state transition for LSTM  
#    Input parameters: X_var_f - time series of size (n_observ_f X n_var_f);
#                      n_var_f - number of response parameters;
#                      n_observ_f - number of time steps;
#    Output parameters: R_covar - matrix of covariance of state transition (n_var_f X n_var_f)
#-----------------------------------------------------------------------
  diff_var=np.zeros((n_var_f,n_observ_f),dtype=np.float32)
  mean_diff_var=np.zeros(n_var_f,dtype=np.float32)
  for i in range(n_var_f):
    for j in range(n_observ_f):
      diff_var[i,j]=np.abs(X_var_f[j,i]-X_Pred_f[j,i]) 
  R_covar=np.zeros((n_var_f,n_var_f), dtype=np.float32)
  for k in range(n_var_f):    
     mean_diff_var[k] = np.mean(diff_var[k,:])  
  for i in range(n_var_f):    
    for j in range(n_observ_f):
      R_covar[i,i]=R_covar[i,i]+(X_var_f[j,i]- mean_diff_var[i])**2
    R_covar[i,i]=R_covar[i,i]/n_observ_f      
  return R_covar

def split_data_overlap_windows(X_data,t_data,n_var_f,n_observ_f,length_window,shift_n,ind_f_start):
  i=1
  #fig, (ax1, ax2) = plt.subplots(2, sharex=True)
  while 1+n_observ_f-(i+1)*length_window+i*shift_n >= 1:
       i=i+1    
  number_windows=i-1
  #print(X_data.shape[0],X_data.shape[1],n_var_f)
  data_array_X_explain = np.zeros([number_windows,length_window,1])
  data_array_X_response = np.zeros([number_windows,length_window])
  data_array_t= np.zeros([length_window,number_windows+1])
 # print(length_window,number_windows,np.size(data_array_t,0),np.size(data_array_t,1))
  #return data_array_X_explain,data_array_X_response,data_array_t
  for i in range(number_windows):
    #print(n_observ_f-1-i*length_window+i*shift_n-(1+n_observ_f-1-(i+1)*length_window+i*shift_n),length_window)
    for j in range(length_window):
      for k in range(1):
      #  print(i,j,n_observ_f-1-(i+1)*length_window-j+(i+1)*shift_n,n_observ_f-1-i*length_window-j+i*shift_n)
        data_array_X_explain[number_windows-i-1,length_window-j-1,k] = \
           X_data[n_observ_f-1-(i+1)*length_window-j+(i+1)*shift_n,0]                                             
        data_array_X_response[number_windows-i-1,length_window-j-1] = \
          X_data[n_observ_f-1-i*length_window-j+i*shift_n,0]
     ## print(i,j,np.size(data_array_t,0),np.size(data_array_t,1))
        data_array_t[length_window-j-1,number_windows-i-1] = \
           t_data[n_observ_f-1-i*length_window-j+i*shift_n]        
  # Find the window preceeding prediction interval of interest. This window should contain start of prediction             
  for i in range(number_windows):
  #  print(t_data[ind_f_start],data_array_t[0,i],data_array_t[length_window-1,i])
    if t_data[ind_f_start]>data_array_t[0,i] and t_data[ind_f_start]<data_array_t[length_window-1,i]:
       ind_window_f=i
    elif t_data[ind_f_start]>data_array_t[length_window-1,i-1] and t_data[ind_f_start]<data_array_t[0,i]:
       ind_window_f=i-1
    elif t_data[ind_f_start]==data_array_t[0,i]:
       ind_window_f=i
    elif t_data[ind_f_start]==data_array_t[length_window-1,i]:
       ind_window_f=i
  tm=dt_ut.convert_sec_date(t_data[ind_f_start])
  t0=dt_ut.convert_sec_date(data_array_t[0,ind_window_f])
  tend=dt_ut.convert_sec_date(data_array_t[length_window-1,ind_window_f])
  print(t0,tend,tm)
 # Definition of last intervals td(1:lengthWindow,numWindows:numWindows+1) corresponding to prediction horizon 
  delt_td=data_array_t[2,0]-data_array_t[1,0] # time step
  for i in range(length_window):
   # print('shift_n',shift_n)
    data_array_t[i,number_windows]=data_array_t[length_window-1-shift_n,number_windows-1]+(i+1)*delt_td
  #  ti_str0=dt_ut.convert_sec_date(data_array_t[i,number_windows])
  #  print(i,shift_n,ti_str0,data_array_t[i,number_windows])
 # for i in range(number_windows):
  #  print(data_array_X_explain[0][0][i],data_array_X_explain[0][0][i])
 # ax1.plot(data_array_t[:,2],data_array_X_explain[2,:,0],color='g',label='Original data')
 # ax1.plot(t_data,X_data[:,0],color='b',label='Original data')
 # xticks = data_array_t[0:length_window-1:30,2]
 # degrees=60
 # date_labels=[]
 # for k in range(length_window):
 #   str_date = dt_ut.convert_sec_date(data_array_t[k,2])
 #   date_labels.append(str_date[0:11])
 # plt.xticks(data_array_t[0:length_window-1:30,2], date_labels[0:length_window-1:30], rotation=degrees)
    #ax0.grid(True)
     # print(number_windows,number_windows-i,\
      #      X_data[1+n_observ_f-1-(i+1)*length_window-j+i*shift_n][2])
   #   print(length_window,len(data_array_t[0:length_window][number_windows-i]),len(t_data[1+n_observ_f-1-(i+1)*length_window+i*shift_n:n_observ_f-1-i*length_window+i*shift_n]))
  #for i in range(number_windows):
  #print(data_array_t[:][number_windows-1])
  #  print(data_array_X_explain[length_window-1][0][number_windows-i])
  #ax1.plot(data_array_X_explain[0:length_window-1][0][number_windows-i-1],data_array_X_explain[0:length_window-1][0][number_windows-i-1],label='Original data')
  #ax1.scatter(data_array_t[:][0],data_array_t[:][0],color='g',label='Original data')
  #ax1.scatter(t_data,t_data,color='g',label='Original data')
  #ax2.scatter(t_data[:],X_data[:][0],label='Original data')  
  #plt.show()
  print(data_array_X_explain.shape)
  return data_array_X_explain, data_array_X_response, data_array_t, number_windows, ind_window_f

def get_median_ensemble(X_ensemble_f,T_ensemble_f,params):
  length_t=np.size(T_ensemble_f,0)
  t_last=np.zeros(length_t,dtype=np.float32)
  X_median = np.zeros(length_t,dtype=np.float32)
  X_std = np.zeros(length_t,dtype=np.float32)
  T_total=[]
  for i in range(length_t):
    t_last[i]=T_ensemble_f[i,0]
  t1={0:t_last[0]}
  for j in range(length_t-1):
    t1.update({j+1: t_last[j+1]})
 # for x, y in thisdict.items():
  for si in range(params['num_shifts']):
   t2=list(T_ensemble_f[:,si])
   lst3 = [key for key in t1 if t1[key] in t2]
   X_collect={str(0):list()}
   X_median_f={str(0):list()}
   X_std_f={str(0):list()}
   for j in range(length_t):
     X_collect.update({str(j):[]})
   for j in range(length_t):
     X_median_f.update({str(j):[]})
     X_std_f.update({str(j):[]})  
   for j in range(length_t):   
     for k in range(len(lst3)): 
      if lst3[k]==j:
        X_values=X_collect[str(j)]
        X_values.append(X_ensemble_f[lst3[k],si])
        X_collect[str(j)]=X_values
   for j in range(length_t):
     X_values=X_collect[str(j)]
     X_values.append(X_ensemble_f[j,si])
     X_collect[str(j)]=X_values
   for j in range(length_t):  
     X_median_f[str(j)]=np.median(X_collect[str(j)])
     X_std_f[str(j)]=np.std(X_collect[str(j)])
     X_median[j] = X_median_f[str(j)]
     X_std[j] = X_std_f[str(j)]
   return X_median, X_std

def un_std(mean_data,sig_data,max_data,X_f_median_f,X_f_std_f,X_true_f,X_downscale_f,ind_responses):
# Unstandartize the data
  nVar_res=X_true_f.shape[1]
  n_observ=X_f_median_f.shape[0]
  X_downscale=np.zeros([nVar_res,n_observ])
  X_true=np.zeros([np.size(X_true_f,0),np.size(X_true_f,1)])
  X_median=np.zeros([n_observ,nVar_res])
  X_std=np.zeros([n_observ,nVar_res])
  for si in range(np.size(X_true_f,1)):
    for j in range(np.size(X_true_f,0)):
      X_true[j,si]=X_true_f[j,si]*max_data[si]+mean_data[si]
    for j in range(n_observ):     
      X_downscale[si,j]=X_downscale[si,j]*max_data[si]+mean_data[si]
      X_median[j,si]=X_f_median_f[j]*max_data[si]+mean_data[si]
      X_std[j]=X_f_std_f[j]*max_data[si]+mean_data[si] 
  return X_median,X_std,X_true,X_downscale 
