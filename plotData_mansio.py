"""----------------------------------------------------------------------------------
   Plot prediction and original data saved in "prediction_data2 directory   
--------------------------------------------------------------------------------------
   

   Last modified Elena 13/08/2020
-------------------------------------------------------------------------------"""

from subsidary import config as cfg
from subsidary import dt_utils as dt_ut
from subsidary import config as cfg
import numpy as np
import json
import csv
from csv import reader
params={}
with open('Summary_prediction'+".json", 'r') as f:
  data=json.load(f)
f.close()
params['session_id']=data['Session ID']
params = cfg.get_params(params)
X_data, t_data, categories_in,categories_out,ind_responses = dt_ut.read_data(params)
categories_in=list(categories_in)
categories_out=list(categories_out)
models_collection =['LSTM','LSTM_EncodeDecode','LSTM_EncodeDecodeCNN','LSTM_EncodeDecodeConv']
params['file_output_prediction']='LSTM'+"_"+params['file_output_prediction']
f = csv.reader(open(params["dir_prediction"]+"/"+params['file_output_history']+".csv", "r"))
i=0
for row in f:
  i=i+1
cols=len(row)
len_d=i
t_true_f=np.zeros(len_d)
X_true_f=np.zeros([len_d,cols-1])
i=0
f = csv.reader(open(params["dir_prediction"]+"/"+params['file_output_history']+".csv", "r"))
for row in f:
    t_true_f[i]=row[0]    
    X_true_f[i,0:2]=row[1:3]
    i=i+1
f = csv.reader(open(params["dir_prediction"]+"/"+params['file_output_prediction']+".csv", "r"))
i=0
for row in f:
  t_m,X_m,X_s=row
  i=i+1
len_d=i
t_prediction=np.zeros(len_d)
X_median=np.zeros([len_d,params['num_models']])
X_std=np.zeros([len_d,params['num_models']])    
for nm in range(params['num_models']):
   params['LSTM_type'] = models_collection[nm]
   params['file_output_prediction'] = 'Output_prediction_'+str(params['session_id'])
   params['file_output_prediction']=params['LSTM_type']+"_"+params['file_output_prediction']

#{'Time start, [s]' : str(t_stamp_prediction[0]), 'Parameter' : name_parameter, 'Units' : ' ml/l ', 'Prediction time' : prediction_interval, \
#          'Maximum value' : str(data_prediction_max), 'Minimum value' : str(data_prediction_min), 'Mean value' : str(data_prediction_mean),\
#          'Network type':str(params['LSTM_type']), 'Prediction horizon, hours': str(params['P_horizon_hours']), 'Number of points for interpolation ': str(params['n_points']), \
#          'Number of inputs': str(params['n_input']), 'Number of outputs': str(params['n_output']), 'Session ID': str(params['session_id'])}
 # open file in read mode
#with open('file.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
 #   csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
  #  for row in csv_reader:
        # row variable is a list that represents a row in csv
   #     print(row)
   i=0
   f = csv.reader(open(params["dir_prediction"]+"/"+params['file_output_prediction']+".csv", "r"))
   for row in f:
    t_m,X_m,X_s=row
    t_prediction[i]=float(t_m)
    X_median[i,nm]=float(X_m)
    X_std[i,nm]=float(X_s)
 #   print(X_median[i,nm],X_std[i,nm],t_prediction[i]) 
    i=i+1  
dt_ut.plot_data(t_prediction,X_median,X_std,t_true_f,X_true_f,categories_out,categories_in,params)
