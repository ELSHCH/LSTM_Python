from math import sqrt
from numpy import split
from numpy import array
import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from subsidary import math_depend as mt_dd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from subsidary.math_depend import sample_covariance as scov
from subsidary.math_depend import covariance_state_transition as scov_trans

def build_model(train_x,train_y,params):
# define parameters
  n_inputs = params['n_input']
# chose model from several LSTM models
  verbose, epochs, batch_size = 0, params['n_epochs'], params['batch_size']
  n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
  model = Sequential()
  if params['LSTM_type'] == 'LSTM':
# define model
   model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
   model.add(Dense(100, activation='relu'))
   model.add(Dense(n_outputs))
   model.compile(loss='mse', optimizer='adam')
  elif params['LSTM_type'] == 'LSTM_EncodeDecode':
   train_y=train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
   model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
   model.add(RepeatVector(n_outputs))
   model.add(LSTM(200, activation='relu', return_sequences=True))
   model.add(TimeDistributed(Dense(100, activation='relu')))
   model.add(TimeDistributed(Dense(1)))
   model.compile(loss='mse', optimizer='adam')
  elif params['LSTM_type'] == 'LSTM_EncodeDecodeCNN':
   train_y=train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
   model.add(Conv1D(filters=64, kernel_size=3, activation='relu', \
                    input_shape=(n_timesteps,n_features)))
   model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
   model.add(MaxPooling1D(pool_size=2))
   model.add(Flatten())
   model.add(RepeatVector(n_outputs))
   model.add(LSTM(300, activation='relu', return_sequences=True))
   model.add(TimeDistributed(Dense(200, activation='relu')))
   model.add(TimeDistributed(Dense(1)))
   model.compile(loss='mse', optimizer='adam')
  elif params['LSTM_type'] == 'LSTM_EncodeDecodeConv':
   n_length = 1
   train_x = train_x.reshape((train_x.shape[0], n_length, 1, n_timesteps, n_features))
# reshape output into [samples, timesteps, features]
   train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
   model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_length, 1, n_timesteps, n_features)))
   model.add(Flatten())
   model.add(RepeatVector(n_outputs))
   model.add(LSTM(100, activation='relu', return_sequences=True))
   model.add(TimeDistributed(Dense(50, activation='relu')))
   model.add(TimeDistributed(Dense(1)))
   model.compile(loss='mse', optimizer='adam')
# fit network
  model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
  return model
def build_modelQRF(train_x,train_y,params):
# define parameters
    n_inputs = params['n_input']
    verbose, epochs, batch_size = 0, params['n_epochs'], params['batch_size']
    n_timesteps, n_features, n_outputs = train_x.shape[1], 1, train_y.shape[1]
# define model
    model = Sequential()
    if params['LSTM_type'] == 'LSTM':
      model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
      model.add(Dense(100, activation='relu'))
      model.add(Dense(n_outputs))
      model.compile(loss='mse', optimizer='adam')
    elif params['LSTM_type'] == 'LSTM_EncodeDecode':
      train_y=train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
      model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
      model.add(RepeatVector(n_outputs))
      model.add(LSTM(200, activation='relu', return_sequences=True))
      model.add(TimeDistributed(Dense(100, activation='relu')))
      model.add(TimeDistributed(Dense(1)))
      model.compile(loss='mse', optimizer='adam')
# fit network
    elif params['LSTM_type'] == 'LSTM_EncodeDecodeCNN':
      train_y=train_y.reshape((train_y.shape[0],train_y.shape[1],1))
      model.add(Conv1D(filters=64,kernel_size=3,activation='relu',\
                    input_shape=(n_timesteps,n_features)))
      model.add(Conv1D(filters=64,kernel_size=3,activation='relu'))
      model.add(MaxPooling1D(pool_size=2))
      model.add(Flatten())
      model.add(RepeatVector(n_outputs))
      model.add(LSTM(300,activation='relu',return_sequences=True))
      model.add(TimeDistributed(Dense(200,activation='relu')))
      model.add(TimeDistributed(Dense(1)))
      model.compile(loss='mse', optimizer='adam')
    elif params['LSTM_type'] == 'LSTM_EncodeDecodeConv':
      n_length = 1
      train_x = train_x.reshape((train_x.shape[0], n_length, 1, n_timesteps, n_features))
# reshape output into [samples, timesteps, features]
      train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
      model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu',\
                           input_shape=(n_length, 1, n_timesteps, n_features)))
      model.add(Flatten())
      model.add(RepeatVector(n_outputs))
      model.add(LSTM(100, activation='relu', return_sequences=True))
      model.add(TimeDistributed(Dense(50, activation='relu')))
      model.add(TimeDistributed(Dense(1)))
      model.compile(loss='mse', optimizer='adam')
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

def time_covariance(network_model, n_var_f, X_response_f, X_explain_f, num_windows, params):
  XTest=np.zeros([params['length_window'],n_var_f])
  R_d=np.zeros(num_windows-1)
  Q_d=np.zeros(num_windows-1)
  R_r=np.zeros(num_windows-1)
  Q_r=np.zeros(num_windows-1)
  Q_std = np.zeros(num_windows)
  R_std = np.zeros(num_windows)
  X_Resp_f=np.zeros([params['length_window'],1])
  X_Pred=np.zeros([params['length_window'],1])
 # fig = plt.figure()
 # ax1 = fig.add_subplot(111)
  n_outputs=1
  for k in range(num_windows-1):
    XTest[0:params['length_window'],0:n_var_f-1]=X_explain_f[k,0:params['length_window'],0:n_var_f-1]
    n_timesteps, n_features = XTest.shape[0], XTest.shape[1]
    X_Resp_f[0:params['length_window'],0]=X_response_f[k,0:params['length_window']]
    if params['LSTM_type'] == 'LSTM':
     input_x = XTest.reshape((1, n_timesteps, n_features))
     X_Pred_f = network_model.predict(input_x, verbose=0)
     X_Pred = np.transpose(X_Pred_f)
 #    ax1.plot(tt[:,k+1],X_Pred)
 #    ax1.plot(tt[:,k],input_x[0,:,0]+1)
    elif params['LSTM_type'] == 'LSTM_EncodeDecode':
     input_x = XTest.reshape((1, n_timesteps, n_features))
     X_Pred_f = network_model.predict(input_x, verbose=0)
     X_Pred[:,0] = X_Pred_f[0,:,0]
    elif params['LSTM_type'] == 'LSTM_EncodeDecodeCNN':
     input_x = XTest.reshape((1, n_timesteps, n_features))
     X_Pred_f = network_model.predict(input_x, verbose=0)
     X_Pred[:,0] = X_Pred_f[0,:,0]
    elif params['LSTM_type'] == 'LSTM_EncodeDecodeConv':
     input_x = XTest.reshape((1, 1, 1, n_timesteps, n_features))
     X_Pred_f = network_model.predict(input_x, verbose=0)
     X_Pred[:,0] = X_Pred_f[0,:,0]
    Q_std[k] = mt_dd.covariance_state_transition(X_Resp_f,X_Pred,params['n_output'],params['length_window']) # covariance of state
    R_std[k] = mt_dd.sample_covariance(X_Resp_f,params['n_output'],params['length_window']) # covariance of time series
  for k in range(num_windows-1):
    R_d[k]=R_std[k]
    Q_d[k]=Q_std[k]
    R_r[k]=R_std[k+1]
    Q_r[k]=Q_std[k+1]
  return Q_d,R_d,Q_r,R_r  

def kalman_corrections(network_model, network_model_KF_Q,network_model_KF_R,\
                       n_var_f, tt,X_explain_f,Q_explain_f,R_explain_f, num_windows,params):
  Ide = np.ones([params['n_output'],params['n_output']])
  F = np.ones([params['n_output'],params['n_output']])
  H = np.ones([params['n_output'],params['n_output']])
  Q_std = 0.0
  R_std = 0.0
  P = 100*np.ones([params['n_output'],params['n_output']])
  Q_res=np.ones([11,1])
  R_res=np.ones([11,1])
  X_Resp =np.zeros([params['length_window'],0])
  z = np.zeros(params['length_window'])
  y = np.zeros(params['length_window'])
 # p_diag=np.zeros([params['length_window'],params['n_output'],i]])
  input_X=np.zeros([1,params['length_window'],n_var_f])
  input_X[0,0:params['length_window'],0:n_var_f-1] = X_explain_f[0:params['length_window'],0:n_var_f-1]
 # input_X[0,0:params['length_window'],1] = X_before_f[0:params['length_window'],n_var_f:3*n_var_f-1]
 # input_X[0,0:params['length_window'],2] = X_before_f[0:params['length_window'],2]
  Predict_f = np.ones([params['length_window'],1])
  X_Pred = np.zeros([params['length_window'],1])
#  R_d=np.zeros([params['length_window'],params['n_output'],num_windows]
#  Q_d=np.zeros([params['length_window'],params['n_output'],num_windows]
#  for k in range(num_windows):
#    XTest=X_response_f[k,0:params['length_window']-1]
#    X_Resp_f=X_explain_f[k,0:params['length_window']-1]
  n_timesteps, n_features, n_outputs = input_X.shape[1], input_X.shape[2], input_X.shape[1]
  if params['LSTM_type'] == 'LSTM_EncodeDecode':
   input_new_x = input_X
  elif params['LSTM_type'] == 'LSTM':
   input_new_x = input_X
  elif params['LSTM_type'] == 'LSTM_EncodeDecodeCNN':
   input_new_x = input_X 
  elif params['LSTM_type'] == 'LSTM_EncodeDecodeConv':
   input_new_x = input_X.reshape((input_X.shape[0], 1, 1, n_timesteps, n_features)) 
# reshape output into [samples, timesteps, features]
  X_Pred_f = network_model.predict(input_new_x, verbose=0)
 
  if params['LSTM_type'] == 'LSTM':
     X_Pred[:,0] = X_Pred_f[0,:]
  elif params['LSTM_type'] == 'LSTM_EncodeDecode':
     X_Pred[:,0] = X_Pred_f[0,:,0]
  elif params['LSTM_type'] == 'LSTM_EncodeDecodeCNN':
     X_Pred[:,0] = X_Pred_f[0,:,0]
  elif params['LSTM_type'] == 'LSTM_EncodeDecodeConv':
     X_Pred[:,0] = X_Pred_f[0,:,0]
  Q_res=predict_lstm_QRF(network_model_KF_Q, Q_explain_f, params)
  R_res=predict_lstm_QRF(network_model_KF_R, R_explain_f, params)
  R_std=R_res[0,0]
  Q_std=Q_res[0,0]
  P = Q_std
#    for i in range(numResponses):
#     for j in range(lengthWindow):
  for i in range(params['length_window']):
    z[i]=X_Pred[i,0]+R_std
    y[i] = z[i] - H[0,0]*X_Pred[i,0]
#  # Estimate Kalman Gain
  K = P*np.transpose(H)*np.linalg.inv(H*P*np.transpose(H)+R_std)
 # predict new state with the Kalman Gain correction
  for k in range(params['length_window']):
   Predict_f[k] = X_Pred[k,0] + K*y[k]
  return Predict_f

def train_lstm_QRF(n_var_f,num_windows,Q_d,R_d,Q_r,R_r,params):
#def train_lstm_QRF(X_response_f,X_explain_f,tt,num_windows,params):
  X_train_f=np.zeros([num_windows,10,1])
  Y_train_f=np.zeros([num_windows,10])
  for k in range(num_windows-1):
    X_train_f[k,0:10,0]= Q_d[k]
    Y_train_f[k,0:10]= Q_r[k]
  network_model_KF_Q = build_modelQRF(X_train_f,Y_train_f,params)
  for k in range(num_windows-1):
    X_train_f[k,0:10,0]= R_d[k]
    Y_train_f[k,0:10]= R_r[k]
  network_model_KF_R = build_modelQRF(X_train_f,Y_train_f,params)
  return network_model_KF_Q, network_model_KF_R

def train_lstm(n_var_f,X_response_f,X_explain_f,tt,num_windows,params):
  X_exp_f=np.zeros([num_windows,params['length_window'],n_var_f])
  X_res_f=np.zeros([num_windows,params['length_window']])
  for k in range(num_windows-1):
    X_exp_f[k,:,0:n_var_f-1]= X_explain_f[k,:,0:n_var_f-1]
    X_res_f[k,:]=X_response_f[k,:]
  network_model = build_model(X_exp_f,X_res_f,params)
  return network_model

def predict_lstm_QRF(network_model_KF, Q_d, params):
  
  input_X=np.zeros([1,10,1])
  input_X[0,0:10,0] = Q_d[0,0]
 # input_X[0,0:params['length_window'],1] = X_before_f[0:params['length_window'],n_var_f:3*n_var_f-1]
 # input_X[0,0:params['length_window'],2] = X_before_f[0:params['length_window'],2]
  
  n_timesteps, n_features, n_outputs = input_X.shape[1], input_X.shape[2], input_X.shape[1]
  if params['LSTM_type'] == 'LSTM_EncodeDecode':
   input_new_x = input_X
  elif params['LSTM_type'] == 'LSTM':
   input_new_x = input_X
  elif params['LSTM_type'] == 'LSTM_EncodeDecodeCNN':
   input_new_x = input_X 
  elif params['LSTM_type'] == 'LSTM_EncodeDecodeConv':
   input_new_x = input_X.reshape((input_X.shape[0], 1, 1, n_timesteps, n_features)) 
# reshape output into [samples, timesteps, features]
  Q_r = network_model_KF.predict(input_new_x, verbose=0)
#  for k in range(num_windows-1):
#    R_r[k,0:params['length_window']-1]=R_std[0,0,k+1]
#    Q_r[k,0:params['length_window']-1]=Q_std[0,0,k+1]
#  for i2 in range(numResponses):
#      R_d(1:lengthWindow,i2,n_r+1)=R_std(i2,i2);
#      Q_d(1:lengthWindow,i2,n_r+1)=Q_std(i2,i2);
  return Q_r
