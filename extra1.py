'''#print(np.size(X_response,0),np.size(X_response,1),X_response.shape)
#Qstd=mt_dd.sample_covariance(X_explain[0,:,:],np.size(X_explain,2),np.size(X_explain,1))
#R_covar=mt_dd.covariance_state_transition(X_explain[0,:,:],X_response[0,:,:],np.size(X_explain,2),np.size(X_explain,1))

 # mt_dd.get_median_ensemble(X_ensemble,[51,61,71,81,91,13,121,12,13,14])
  X_exp = np.zeros([number_windows,params['length_window'],1])
  X_res = np.zeros([number_windows,params['length_window']])
  Q_exp = np.zeros([number_windows,params['length_window'],1])
  R_exp = np.zeros([number_windows,params['length_window'],1])
  Q_res = np.zeros([number_windows,params['length_window']])
  R_res = np.zeros([number_windows,params['length_window']])
  X_exp[0:number_windows,0:params['length_window'],0]=\
  X_explain[0:number_windows,0:params['length_window'],0]
  X_res[0:number_windows,0:params['length_window']]=X_response[0:number_windows,0:params['length_window']]
#  print('ind_window',ind_window,data_array_t[0,ind_window],data_array_t[params['length_window'],ind_window])
#  t1=dt_ut.convert_sec_date(data_array_t[0,ind_window])
#  t2=dt_ut.convert_sec_date(data_array_t[params['length_window'],ind_window])
#  t0=dt_ut.convert_sec_date(t_downscale_f[ind_start])
#  print(t1,t0,t2)
  X_before = X_explain[ind_window,:,0]              
  if network_exists[i] == 0:
    model = m_lstm_kf.build_model(X_exp,X_res,params)
 #model_KF=m_lstm_kf.build_model(X_explain,X_response,params)
    dt_ut.save_network_json(model,network_names[i],params)
 #save_network_json(model_KF,network_names_KF[i])
  else:
    model=dt_ut.load_network_json(network_names[i],params)
 #model_KF=load_network_json(network_names_KF[i])
#X_before = np.array([np.size(X_explain,1),np.size(X_explain,2)])
#X_before=X_explain[2,:,0]
  if network_KF_exists[i] == 0:      
     Q_exp,R_exp,Q_res,R_res = m_lstm_kf.kalman_corrections(model, X_res, X_exp, number_windows, params)
     model_KF = m_lstm_kf.train_lstm_QRF(X_res,X_exp,number_windows,\
                                         Q_exp,R_exp,Q_res,R_res,params)
  else: 
     X_predictions = m_lstm_kf.predict_lstm_QRF(model_KF, X_before, params)
  X_ensemble[0:params['length_window'],i]=X_predictions[:]
  t_ensemble[0:params['length_window'],i]=data_array_t[:,number_windows]
  #if i>0:                 
  #t_ensemblе[0:params['length_window']-1,i]=data_array_t[:,number_windows]                
X_f_median, X_f_std = mt_dd.get_median_ensemble(X_ensemble,t_ensemble,params)  
#print(predictions)
#fig, (ax1, ax2) = plt.subplots(2, sharex=True)
#ax1.plot(data_array_t[:][i],data_array_X_explain[:][0][i],label='Original data')
#ax1.plot(t_downscale_f[:],X_downscale_f[:][0],label='Original data')
#ax1.legend((categories_in[0], 'Predicted data'),
     #      loc='lower left')
t_prediction=np.zeros(params['length_window'])
for i in range(params['length_window']):
 t_prediction[i] =  t_ensemble[i,0]
X_median,X_std,X_true,X_filter = \
                mt_dd.un_std(mean_data,sig_data,max_data,X_f_median,X_f_std,X_true_f,X_downscale_f,ind_responses)
dt_ut.save_prediction(categories_out[0], params['P_horizon_hours'],t_prediction,X_f_median, \
                    t_downscale_f,X_downscale_f,params)
dt_ut.plot_data(t_prediction,X_median,X_std,t_true_f,X_true_f,categories_out,categories_in,params)

#print(len(t_downscale_f[:]),len(X_downscale_f[:][0]))
#ax2.plot(t_downscale_f[:],X_f_mean[0][:],label=categories_in[0])
#plt.show()                                                                                         
# Save data and network in separate higher level directories
save_prediction(name_parameter,prediction_interval,t_stamp_prediction,X_prediction, \
                    t_stamp_history,X_history,params)
plot_data(t_f,X_f,t_true,X_true,categories_f,categories_true,params)                

