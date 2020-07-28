from math import sqrt
from numpy import split
from numpy import array
import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from subsidary import math_depend as mt_dd
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from subsidary.math_depend import sample_covariance as scov
from subsidary.math_depend import covariance_state_transition as scov_trans

def build_model(train_x,train_y,params):
	# define parameters
	n_inputs = params['n_input']
	verbose, epochs, batch_size = 0, params['n_epochs'], params['batch_size']
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# define model
	model = Sequential()
	model.add(LSTM(400, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(Dense(300, activation='relu'))
	model.add(Dense(n_outputs))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model
def build_modelQRF(train_x,train_y,params):
	# define parameters
	n_inputs = params['n_input']
	verbose, epochs, batch_size = 0, params['n_epochs'], params['batch_size']
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# define model
	model = Sequential()
	model.add(LSTM(400, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(Dense(300, activation='relu'))
	model.add(Dense(n_outputs))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model
    # make a forecast
    # make a forecast
def forecast(model, input_x, n_input):
 # reshape into [1, n_input, 1]
 input_x = input_x.reshape((1, len(input_x), input_x.shape[1]))
 # forecast the next time period
 yhat = model.predict(input_x, verbose=0)
 # we only want the vector forecast
 yhat = yhat[0]
 return yhat

def kalman_corrections(network_model, X_response_f, X_explain_f,num_windows,params): 
  XTest=np.zeros([params['length_window'],1])
  R_d=np.zeros([num_windows-1,params['length_window'],params['n_output']])
  Q_d=np.zeros([num_windows-1,params['length_window'],params['n_output']])
  R_r=np.zeros([num_windows-1,params['length_window']])
  Q_r=np.zeros([num_windows-1,params['length_window']])  
  Q_std = np.zeros([params['n_output'],params['n_output'],num_windows])
  R_std = np.zeros([params['n_output'],params['n_output'],num_windows])
  X_Resp_f=np.zeros([params['length_window'],1])
  for k in range(num_windows): 
    XTest[0:params['length_window'],0]=X_explain_f[k,0:params['length_window'],0]
    X_Resp_f[0:params['length_window'],0]=X_response_f[k,0:params['length_window']]
    input_x = XTest.reshape((1, len(XTest), 1))
    X_Pred_f = network_model.predict(input_x, verbose=0)
    X_Pred_f = np.transpose(X_Pred_f)
    Q_std[0,0,k] = mt_dd.covariance_state_transition(X_Resp_f,X_Pred_f,params['n_output'],params['length_window']) # covariance of state       
    R_std[0,0,k] = mt_dd.sample_covariance(X_Resp_f,params['n_output'],params['length_window']) # covariance of time series 
  for k in range(num_windows-1):  
    R_d[k,0:params['length_window'],0]=R_std[0,0,k]
    Q_d[k,0:params['length_window'],0]=Q_std[0,0,k]
    R_r[k,0:params['length_window']]=R_std[0,0,k+1]
    Q_r[k,0:params['length_window']]=Q_std[0,0,k+1]
  return Q_d,R_d,Q_r,R_r

def train_lstm_QRF(X_response_f,X_explain_f,num_windows,Q_d,R_d,Q_r,R_r,params):
  X_exp_f=np.zeros([num_windows,3*params['length_window'],1])
  X_res_f=np.zeros([num_windows,3*params['length_window']])
  for k in range(num_windows-1):       
    X_exp_f[k,0:3*params['length_window'],0]=np.concatenate((X_explain_f[k,0:params['length_window'],0],\
                                    Q_d[k,0:params['length_window'],0], \
                                    R_d[k,0:params['length_window'],0]), axis=0)
    X_res_f[k,0:3*params['length_window']]=np.concatenate((X_response_f[k,0:params['length_window']],\
                                    Q_r[k,0:params['length_window']], \
                                    R_r[k,0:params['length_window']]), axis=0)
  network_model_KF = build_modelQRF(X_exp_f,X_res_f,params)
  return network_model_KF

def predict_lstm_QRF(network_model_KF,X_before_f,params):
  Ide = np.ones([params['n_output'],params['n_output']])
  F = np.ones([params['n_output'],params['n_output']]) 
  H = np.ones([params['n_output'],params['n_output']])
  Q_std = np.ones([params['n_output'],params['n_output']])
  R_std = np.ones([params['n_output'],params['n_output']]) 
  P = 100*np.ones([params['n_output'],params['n_output']])
  z = np.zeros(params['length_window'])
  y = np.zeros(params['length_window'])
 # p_diag=np.zeros([params['length_window'],params['n_output'],i]])
  XTest=np.zeros([3*params['length_window'],1])
  XTest[0:3*params['length_window'],0]=X_before_f[0:3*params['length_window']]
  print('XTest',XTest)
  input_X = XTest.reshape((1, np.size(XTest,0), 1))
  Predict_f = np.ones(params['length_window']) 
#  R_d=np.zeros([params['length_window'],params['n_output'],num_windows]
#  Q_d=np.zeros([params['length_window'],params['n_output'],num_windows]      
#for k in range(num_windows): 
#    XTest=X_response_f[k,0:params['length_window']-1]
#    X_Resp_f=X_explain_f[k,0:params['length_window']-1]
  X_Pred_f = network_model_KF.predict(input_X, verbose=0)
  print('X_Pred_f',X_Pred_f)
 # Q_std = mt_dd.covariance_state_transition(X_Resp_f,X_Pred_f,params['n_output'],params['length_window']) # covariance of state       
#  R_std = mt_dd.sample_covariance(X_Pred_f,params['n_output'],params['length_window']) # covariance of time series 
#    for i2 in range(params['n_output']):
#      R_d[0:params['length_window']-1,i2,k]=R_std[i2,i2]
#      Q_d[0:params['length_window']-1,i2,k]=Q_std[i2,i2]               
#       YPred = predict(netQRF,XTest,'MiniBatchSize',10);
  X_Pred_f  = np.transpose(X_Pred_f)
  for i1 in range(params['n_output']):
    Q_std[i1,i1] = X_Pred_f[2*params['length_window'],0]
    R_std[i1,i1] = X_Pred_f[2*params['length_window'],0] 
  P = Q_std 
#    for i in range(numResponses):  
#     for j in range(lengthWindow):
  for i in range(params['length_window']):
    z[i]=X_Pred_f[params['length_window']+i,0]+R_std[0,0]
    y[i] = z[i] - H[0,0]*X_Pred_f[params['length_window']+i,0]
#  # Estimate Kalman Gain
  K = P*np.transpose(H)*np.linalg.inv(H*P*np.transpose(H)+R_std)
 # predict new state with the Kalman Gain correction
  Predict_f[0:params['length_window']] = X_Pred_f[0:params['length_window'],0] + \
                                         K*y[0:params['length_window']]
#    P = (Ide - K*H)*P*np.transpose(Ide - K*H) + K*R_std*np.transpose(K)
# if n_r==numWindows-1
# Q_std = covarianceStateTransitionPattern(xd(1:lengthWindow,1:numResponses,n_r+1)',\
#         XPred(1:numResponses,1:lengthWindow),numResponses,lengthWindow);
#    for i1=1:numResponses
#      xq(i1,1:lengthWindow)=xd(1:lengthWindow,i1,n_r+1); % add uncertainty for flexible predicton
#    R_std = sampleCovariance(xq',numResponses,lengthWindow); % covariance of time series 
#
#  for i2 in range(numResponses): 
#      R_d(1:lengthWindow,i2,n_r+1)=R_std(i2,i2);
#      Q_d(1:lengthWindow,i2,n_r+1)=Q_std(i2,i2);
  return Predict_f     


