import os
import platform
import getpass
import locale

#locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')

def get_params(params):
   
#Set network hyperparameters 
   params['nlayer']= 3 # number of layers for recurrent neural network f - model
   params['Qnlayer'] = 1  # number of layers for recurrent neural network Q - model 
   params['Rnlayer'] = 1  # number of layers for recurrent neural network R - model
   params['Knlayer'] = 3  # number of layers for recurrent neural network Kalman Gain - model
   params['Flayer'] = 1  # number of layers for recurrent neural network F - model
   params['n_output']= None # number of output variables
   params['n_hidden']= 512 # LSTM hidden number of layers for f - LSTM model 
   params['Qn_hidden']= 256 # LSTM hidden number of layers for Q-LSTM model
   params['Rn_hidden']= 256 # LSTM hidden number of layers for R-LSTM model
   params['Kn_hidden']= 128 # LSTM hidden number of layers for Kalman Gain-LSTM model
   params['P_mul']= 0.1
   params['K_inp']=48
   params['n_input']= None#  number of input variables
   params['length_window'] = None # length of trained sequence
   params['num_hidden']=20
   params['n_epochs']=30
   params['batch_size']=10
   params['num_models'] = 4

   params['per_process_gpu_memory_fraction']=1

   #Set data directories 
   
   params["dir_model"]='C:/Users/eshchekinova/Documents/BoknisData/LSTMPython'
   wd = params["dir_model"]
   params["data_input"]=wd+'/InPrediction.dat'
   params["list_categories"]=wd+'/ListCategories.dat'
   params["list_categories_init"]=wd+'/InitialCategories.dat'
   params["dir_network"]=wd+'/LSTM_network'
   params["dir_prediction"]=wd+'/prediction_data'
   params["dir_LSTM_models"]=wd+'/model_list'
   params["dir_utils"]=wd+'/subsidary'


   # Read parameters from prediction file
   fl = open(params['data_input'],"r")          
   params['time_start'] = fl.readline().rstrip()
   params['P_horizon_hours']= int(fl.readline())
   params['n_points'] = int(fl.readline())   
   params['nVar'] = int(fl.readline())
   line_file_name = fl.readline()
   params['file_data'] =line_file_name.rstrip()
   params['file_summary_prediction'] = 'Summary_prediction'
   params['file_output_prediction'] = 'Output_prediction_'+str(params['session_id'])
   params['file_output_history'] = 'Output_history_'+str(params['session_id'])
   params['PredictionPlot'] = 'PredictionPlot_'+str(params['session_id'])

 #  params['file_prediction']=params['']+"/files/logs/"+params["model"]+"_"+params["rn_id"]+"_"+str(params['run_mode'])+"_"+utils.get_time()+".txt"
   return params
def update_parameters(number_var_in,number_var_out,train_int_length,params):
   # Update parameters for prediction
   params['file_output_prediction'] = 'Output_prediction_'+str(params['session_id'])
   params['length_window']= train_int_length
   params['n_output']= number_var_out
   params['n_input']= number_var_in
   params['file_network']= params['LSTM_type']+ "_" + \
                              str(params['P_horizon_hours']) + "_" +\
                              str(params['n_points'])+"_"+ \
                              str(params['n_input']) + "_" + str(params['n_output']) + "_" + \
                              str(params['num_hidden'])                                   
   params['file_network_KF_Q']=params['LSTM_type']+"_QF"+ "_" +\
                              str(params['P_horizon_hours'])+"_"+str(params['n_points'])+"_"+ \
                              str(params['n_input']) + "_" + str(params['n_output']) + "_" + \
                              str(params['num_hidden'])
   params['file_network_KF_R']=params['LSTM_type']+"_RF"+ "_" +\
                              str(params['P_horizon_hours'])+"_"+str(params['n_points'])+"_"+ \
                              str(params['n_input']) + "_" + str(params['n_output']) + "_" + \
                              str(params['num_hidden'])    
   return params
 
   
def update_categories(): 
   # Read list of initial categories
   categories_init = list()
   with open(params["list_categories_init"],'r') as f2:
      for line in f2: 
        [categ] = line.split()
        categories_init.append(categ) 
   f2.close()        
   # Read list of input categories - parameters
   categories_def = list()
   key_cat = list()
   key_out = list()
   with open(params["list_categories"],'r') as f3:
      for line in f3: 
        [value_in, value_out, key_c]=line.split()
        if int(value_in) == 1: key_cat.append(key_c)
        if int(value_out) == 1: key_out.append(key_c)
   f3.close()  
   categories_def = key_cat
   categories_responses = key_out 
   return categories_init,categories_def,categories_responses   
