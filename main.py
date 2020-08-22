"""----------------------------------------------------------------------------------
  Algorithm of prediction of multiple time series using Long-Short Term Memory Networks 
--------------------------------------------------------------------------------------

    LSTM algorithm is implemented with three schemes :
    "3LSTM_PATTERN_KF" and "LSTM_PATTERN_KF" and "LSTM_PATTERN".
    In "LSTM_PATTERN" scheme a standard sequence-to-sequence forward prediction by
    LSTM is used;
    in the "LSTM_PATTERN_KF" scheme a standard sequence-to-sequence forward prediction
    is used together with following correction step using Kalman Gain
    in the "3LSTM_PATTERN_KF" scheme the LSTM prediction data are updated using
    Kalman Gain, however, Kalman Gain is evaluated based on LSTM prediction of
    model and data covariance uncertainties 

    Two options for prediction could be used 1)"INTERVAL"- interval of length
    "numpoints" could be used for making step forward prediction at each time step
    2) "POINT" only last data point is used for prediction of next point
    
    The network could be trained using entire time series or selected
    interval, therefore two options for choice of training : 'FULL', 'PART'

    Initial parameters are set in input file 'InputPrediction.dat'

   Last modified Elena 26/11/2019
-------------------------------------------------------------------------------"""
import sys,os
from subsidary import config as cfg
from subsidary import dt_utils as dt_ut
from subsidary import math_depend as mt_dd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from model_list import multistep_lstm_kf as m_lstm_kf
from datetime import datetime
import time

if __name__=="__main__":

 global params
 params={}   
 now=time.localtime()
 t=time.mktime(now)
 time_str=str(t)
 params['session_id'] = time_str[1:-2] 
 # Initialize key parameters 
 params = cfg.get_params(params)
 # Set the categories of parameters: categories_init - list of visible parameters,
 #                                   categories_def - list of predictor parameters selected by user,
 #                                   categories_responses - list of responses
 # categories_def and categories_predictors might not coincide completely 
 #categories_init,categories_def,categories_responses=cfg.update_categories()
 # Read original data from.dat file
 X_data, t_data, categories_in,categories_out,ind_responses = dt_ut.read_data(params)
 time_start_sec = dt_ut.convert_to_secs(params['time_start'])
 '''X_history=np.array(X_data[0][:])
 X_prediction=np.array(X_data[0][0:20])
 t=np.array(t_data) 
 X_history=np.transpose(X_history)
 X_prediction=np.transpose(X_prediction)'''
 ### Prepare time series: standartized time series, define downsampled time series and 
 ###  build training sequences for training network model
 t_true_f,X_true_f,t_downscale_f,X_downscale_f,XTrain_f,YTrain_f,sample_size_f,mean_data,sig_data,max_data = \
                            dt_ut.prepare_data(X_data,t_data,params)
 X_downscale_f=X_downscale_f.transpose()
 X_true_f=np.array(X_true_f).transpose()
 print('X_true',X_true_f.shape,X_downscale_f.shape)
 # Check if start time is inside the time series range
 message_exit=dt_ut.check_time_start(time_start_sec,t_downscale_f)
 if message_exit:
  print(message_exit)
  sys.exit()
 
 #Find begining and end of prediction interval in the downsampled time series
 ind_start,ind_end,sample_size_d,window_size_d = \
                            dt_ut.define_initerval_time(sample_size_f,time_start_sec,t_downscale_f,params)
 n_var_in = len(categories_in)
 n_var_out = len(categories_out)
 categories_in=list(categories_in)
 categories_out=list(categories_out)
# Define shift of overlapping windows of training data
 params['num_shifts'] = 6 # number of shifts of overlapped trained sequences
 shift_size = np.zeros((params['num_shifts']))
 for i in range(params['num_shifts']):
  shift_size[i]=2*i
 #shift_size[2]=sample_size_d-1
 shift_size=shift_size.astype(int)
 models_collection =['LSTM','LSTM_EncodeDecode','LSTM_EncodeDecodeCNN','LSTM_EncodeDecodeConv']
 params['LSTM_type'] = models_collection[0]
 params=cfg.update_parameters(n_var_in,n_var_out,sample_size_d,params)
 data_prediction_mean=np.zeros(params['num_models'])
 data_prediction_max=np.zeros(params['num_models'])
 data_prediction_min=np.zeros(params['num_models'])
 t_prediction=np.zeros(params['length_window'])
 fig = plt.figure()
 ax1 = fig.add_subplot(111)
 
 # Select prediction model
 for nm in range(params['num_models']):    
   params['LSTM_type'] = models_collection[nm]
   params=cfg.update_parameters(n_var_in,n_var_out,sample_size_d,params)
   params['file_summary_prediction']=params['file_summary_prediction']
   params['file_output_prediction']=params['LSTM_type']+"_"+params['file_output_prediction']
 #Define network names according to selected user initial parameters
   network_names_KF_Q,network_names_KF_R,network_names =list(),list(),list()
   for i in range(params['num_shifts']):
    name_networkKF_Q,name_networkKF_R,name_network=dt_ut.set_network_name(shift_size[i],params)
    network_names_KF_Q.append(name_networkKF_Q)
    network_names_KF_R.append(name_networkKF_R)
    network_names.append(name_network)
 # Check condition if LSTM pretrained network does already exist
   network_exists,network_KF_Q_exists,network_KF_R_exists =[0 for x in range(params['num_shifts'])],\
                                   [0 for x in range(params['num_shifts'])],[0 for x in range(params['num_shifts'])]
   network_exists,network_KF_Q_exists,network_KF_R_exists = \
                        dt_ut.find_name_network(network_names_KF_Q,network_names_KF_R,network_names,shift_size,params)

 # Run prediction or training mode depending on condition network exist or not
 ##t_prediction = t_downscale_f[-500:]
 ##X_prediction =X_downscale_f[-500:][0]
 ##t_history = t_downscale_f
 ##X_history = X_downscale_f[:][0]
 ##dt_ut.save_prediction("Oxygen",48,t_prediction,X_prediction, \
                ##    t_history,X_history,params)
 #dt_ut.run_model(params,network_exists,network_KF_exists,shift_size,...)
 #X_ensemble=np.zeros([params['length_window'],params['num_shifts']])
 #t_ensemble=np.zeros([params['length_window'],params['num_shifts']])
   X_predictions=np.zeros(params['length_window'])
   n_observ_f=X_downscale_f.shape[0]
   X_before = np.zeros([params['length_window'],3*n_var_in])
   X_check = np.zeros([params['length_window']])
   X_ensemble=np.zeros([params['length_window'],params['num_shifts']])
   t_ensemble=np.zeros([params['length_window'],params['num_shifts']])
   X_f_median=np.zeros([params['length_window'],0])
   X_f_std =np.zeros([params['length_window'],0])
   for i in range(params['num_shifts']):
     X_explain,X_response,data_array_t,number_windows,ind_window=mt_dd.split_data_overlap_windows(X_downscale_f,t_downscale_f,\
                          n_var_in,n_observ_f,params['length_window'],shift_size[i],ind_start)
   #print(np.size(X_explain,0),np.size(X_explain,1),np.size(X_explain,2))
 #for i in range(number_windows):  
 #  ax.plot(data_array_t[:,i],X_explain[i,:,0],'o')
 #for i in range(number_windows):   
 #  ax.plot(data_array_t[:,i+1],X_response[i,:]+0.1,'x')
 #  ax.plot(t_downscale_f,X_downscale_f[:,0])
 # # ax.plot(X,Y2,'x')
 #plt.show()  
 #print(np.size(X_response,0),np.size(X_response,1),X_response.shape)
 #Qstd=mt_dd.sample_covariance(X_explain[0,:,:],np.size(X_explain,2),np.size(X_explain,1))
 #R_covar=mt_dd.covariance_state_transition(X_explain[0,:,:],X_response[0,:,:],np.size(X_explain,2),np.size(X_explain,1))

 # mt_dd.get_median_ensemble(X_ensemble,[51,61,71,81,91,13,121,12,13,14])
     X_exp = np.zeros([number_windows,params['length_window'],n_var_in])
     X_res = np.zeros([number_windows,params['length_window']])
     Q_exp = np.zeros(number_windows)
     R_exp = np.zeros(number_windows)
     Q_res = np.zeros(number_windows)
     R_res = np.zeros(number_windows)
     Q_before = np.zeros([10,1])
     R_before = np.zeros([10,1])
     X_exp[0:number_windows,0:params['length_window'],0:n_var_in-1]=\
        X_explain[0:number_windows,0:params['length_window'],0:n_var_in-1]
     X_res[0:number_windows,0:params['length_window']]=X_response[0:number_windows,0:params['length_window']]
#  print('ind_window',ind_window,data_array_t[0,ind_window],data_array_t[params['length_window'],ind_window])
#  t1=dt_ut.convert_sec_date(data_array_t[0,ind_window])
#  t2=dt_ut.convert_sec_date(data_array_t[params['length_window'],ind_window])
#  t0=dt_ut.convert_sec_date(t_downscale_f[ind_start])
#  print(t1,t0,t2)
     if network_exists[i] == 0:
       model = m_lstm_kf.train_lstm(n_var_in,X_res,X_exp,data_array_t,number_windows,params)
 #model_KF=m_lstm_kf.build_model(X_explain,X_response,params)
       dt_ut.save_network_json(model,network_names[i],params)
   # dt_ut.save_network_json(model_KF,network_names_KF[i],params)
     else:
       model=dt_ut.load_network_json(network_names[i],params)
  #  model_KF=dt_ut.load_network_json(network_names_KF[i],params)
#X_before = np.array([np.size(X_explain,1),np.size(X_explain,2)])
#X_before=X_explain[2,:,0]
     if network_KF_Q_exists[i] == 0 and network_KF_R_exists[i] == 0:
       Q_exp,R_exp,Q_res,R_res = m_lstm_kf.time_covariance(model,n_var_in, X_res, X_exp, number_windows, params)
       model_KF_Q, model_KF_R = m_lstm_kf.train_lstm_QRF(n_var_in,number_windows,\
                                         Q_exp,R_exp,Q_res,R_res,params)
     # model_KF = m_lstm_kf.train_lstm(X_res,X_exp,data_array_t,number_windows,params)
       dt_ut.save_network_json(model_KF_Q,network_names_KF_Q[i],params)
       dt_ut.save_network_json(model_KF_R,network_names_KF_R[i],params)
     else:
  #   for j in range(number_windows):
  #     plt.plot(data_array_t[:,j],X_exp[j,:,0])
  #     plt.plot(data_array_t[:,j],X_res[j,:]+1)
       model_KF_Q=dt_ut.load_network_json(network_names_KF_Q[i],params)
       model_KF_R=dt_ut.load_network_json(network_names_KF_R[i],params)
       Q_exp,R_exp,Q_res,R_res = m_lstm_kf.time_covariance(model,n_var_in, X_res, X_exp, number_windows, params)
      # for s1 in range(number_windows-1):
      #  X_before = np.concatenate((X_exp[ind_window,0:params['length_window'],0],\
      #                              Q_exp[ind_window,0:params['length_window'],0], \
      #                              R_exp[ind_window,0:params['length_window'],0]), axis=0)  
      #  X_predictions = m_lstm_kf.predict_lstm_QRF(model_KF, X_before, params)
       X_before[0:params['length_window'],0:n_var_in-1] = X_exp[ind_window,0:params['length_window'],0:n_var_in-1]
       Q_before[0:10,0] = Q_exp[ind_window]
       R_before[0:10,0] = R_exp[ind_window]
      #  X_before[0:params['length_window'],0:n_var_in-1] = X_exp[s1,0:params['length_window'],0:n_var_in-1]
      #  Q_before[0:10,0] = Q_exp[s1]
      #  R_before[0:10,0] = R_exp[s1] 
      #  X_check[0:params['length_window']] = X_res[ind_window,0:params['length_window']]
       X_predictions = m_lstm_kf.kalman_corrections(model,model_KF_Q,model_KF_R, n_var_in,data_array_t,\
                                            X_before,Q_before,R_before, number_windows, params)
  
    #input_X = X_before.reshape((1, len(X_before), 1))
    #X_predictions = model_KF.predict(input_X, verbose=0)
    # for s1 in range(number_windows-1):       
     #   ax1.plot(data_array_t[:,s1],X_exp[s1,0:params['length_window'],0])
     #   ax1.plot(data_array_t[:,s1+1],X_predictions[0:params['length_window']])
      #  X_ensemble[0:params['length_window'],s1]=X_exp[s1,0:params['length_window'],0]
      #  t_ensemble[0:params['length_window'],s1]=data_array_t[:,s1] 
      #  print('X_before',X_predictions)
     for s1 in range(params['length_window']):   
        X_ensemble[s1,i]=X_predictions[s1]
 #  fig.show()
  # X_ensemble[0:params['length_window'],i]=X_before[0:params['length_window'],0]
        t_ensemble[s1,i]=data_array_t[s1,ind_window+1]
        dt_object = datetime.fromtimestamp(t_ensemble[s1,i])
        str_time = dt_object.strftime("%d-%b-%Y (%H:%M:%S)")
        print(str_time)
 #  print('X_before',X_before) 
   #if i>0:                 
   #t_ensemblе[0:params['length_window']-1,i]=data_array_t[:,number_windows]
# t_before = np.zeros(params['length_window'])
# t_before = data_array_t[:,ind_window]
   X_f_median, X_f_std = mt_dd.get_median_ensemble(X_ensemble,t_ensemble,params)
  # ax1.plot(t_ensemble[:,0],X_ensemble[:,0],'b')
  # ax1.plot(t_ensemble[:,1],X_ensemble[:,1],'k')
  # ax1.plot(t_ensemble[:,0],X_f_median,'r')
  # ax1.plot(t_ensemble[:,0],X_f_std,'ro')
   for i in range(params['length_window']):
     t_prediction[i] =  t_ensemble[i,0]
   X_median,X_std,X_true,X_filter = \
                mt_dd.un_std(mean_data,sig_data,max_data,X_f_median,X_f_std,X_true_f,X_downscale_f,ind_responses)
   dt_ut.save_prediction(n_var_in,t_prediction,X_median, X_std,t_downscale_f,X_filter,params)
   data_prediction_mean[nm]=np.mean(X_median[:,0])
   data_prediction_max[nm]=np.max(X_median[:,0])
   data_prediction_min[nm]=np.min(X_median[:,0])
 data_max=np.max(X_filter[:,0])
 data_min=np.min(X_filter[:,0])
 data_mean=np.mean(X_filter[:,0])
 dt_ut.save_summary_prediction(categories_out[0], params['P_horizon_hours'],t_prediction, \
                            data_prediction_mean, data_prediction_max, data_prediction_min,\
                            data_max, data_min, data_mean, params)    
 fig.show()
# dt_ut.plot_data(t_prediction,X_median,X_std,t_true_f,X_true,categories_out,categories_in,params)
 #plt.plot(t_ensemble[:,1],t_ensemble[:,1]+t_ensemble[:,0]/10,'o')
 #plt.plot(t_ensemble[:,2],t_ensemble[:,2]+2*t_ensemble[:,0]/10,'bo')
 #plt.show()
 #for i in range(params['length_window']):
 #  ti_str0=dt_ut.convert_sec_date(t_ensemble[i,0])
 #  print(i,ti_str0)
 #for i in range(params['length_window']):
 #  ti_str1=dt_ut.convert_sec_date(t_ensemble[i,1])
 #  print(ti_str1)
 #for i in range(params['length_window']):
 #  ti_str2=dt_ut.convert_sec_date(t_ensemble[i,2])
#  print(ti_str2)   
 # ti_str1[i]=dt_ut.convert_sec_date(t_ensemble[1,i])
 # ti_str2[i]=dt_ut.convert_sec_date(t_ensemble[2,i])

