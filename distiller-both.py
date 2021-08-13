'''
Created by: Jose Guadalupe Hernandez
Email: jgh9094@gmail.com

This file will distill the knowledge from teachers into a Multi-task CNN.
Will only deal with softmax outputs and transform student logit output
'''

# general python imports
import numpy as np
import argparse
import os
import pandas as pd

# keras python inputs
from keras.models import Model
from keras.layers import Input,Embedding,Dropout,Dense,GlobalMaxPooling1D,Conv1D
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy as logloss
from keras import backend as K
from keras import initializers
from keras.layers.merge import Concatenate
from sklearn.metrics import f1_score

# summit specific imports
from loaddata6reg import loadAllTasks
from mpi4py import MPI

# global variables
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.size #Node count. size-1 = max rank.
EPOCHS = 100
CLASS =  [4,639,7,70,326]
TEMP = 0

# return configuration for the experiment
def GetModelConfig(config):
  # testing configuration
  if config == 0:
    return {
      'learning_rate': 0.01,
      'batch_size': 256,
      'dropout': 0.5,
      'optimizer': 'adam',
      'wv_len': 300,
      'emb_l2': 0.001,
      'in_seq_len': 1500,
      'filter_sizes': [3,4,5],
      'num_filters': [300,300,300],
      'temp': [1,2,5,7,10,13,15,17,20,22,25,30]
    }
  if config == 1:
    return {
      'learning_rate': 0.01,
      'batch_size': 256,
      'dropout': 0.5,
      'optimizer': 'adam',
      'wv_len': 300,
      'emb_l2': 0.001,
      'in_seq_len': 1500,
      'filter_sizes': [3,4,5],
      'num_filters': [300,300,300],
      'temp': [0.1,0.2,0.5,0.7,0.9,1.0,1.1,1.2,1.5,1.7,1.9,2.0]
    }
  else:
    print('MODEL CONFIGURATION DOES NOT EXIST')
    exit(-1)

# softmax/temperture output transformer
def softmax(x,t):
    ex = np.exp(x/t)
    tot = np.sum(ex)
    return np.array(ex / tot)

# Y data is divided by temperature and softmax applied
def LoadY(teach,temp):
  print('CONCAT DATA:', flush= True)
  Y,YV = [],[]
  # iterate through the number of classes in the training data
  for i in range(len(CLASS)):
    print(str(i), flush= True)
    # get training dir
    Y.append([])
    yt = np.load(teach + 'training-task-' + str(i) + '.npy')
    # transform the teacher data
    for j in range(yt.shape[0]):
      Y[i].append(softmax(yt[j], temp))
    # make a numpy array
    Y[i] = np.array(Y[i])


    # get validation dir
    YV.append([])
    yvt = np.load(teach + 'validating-task-' + str(i) + '.npy')
    # transform the teacher data
    for j in range(yvt.shape[0]):
      YV[i].append(softmax(yvt[j], temp))
    YV[i] = np.array(YV[i])

  print('Training Output Data', flush= True)
  i = 0
  for y in Y:
    print('task', i, flush= True)
    print('--cases:', len(y), flush= True)
    print('--classes:',len(y[0]), flush= True)
    i += 1
  print()

  print('Validation Output Data', flush= True)
  i = 0
  for y in YV:
    print('task', i, flush= True)
    print('--cases:', len(y), flush= True)
    print('--classes:',len(y[0]), flush= True)
    i += 1
  print()

  return Y,YV

# will return a mt-cnn with a certain configuration
# output will by raw logits
def CreateMTCnn(num_classes,vocab_size,cfg):
    # define network layers
    input_shape = tuple([cfg['in_seq_len']])
    model_input = Input(shape=input_shape, name= "Input")
    # embedding lookup
    emb_lookup = Embedding(vocab_size, cfg['wv_len'], input_length=cfg['in_seq_len'],
                            embeddings_initializer= initializers.RandomUniform( minval= 0, maxval= 0.01 ),
                            name="embedding")(model_input)
    # convolutional layer and dropout
    conv_blocks = []
    for ith_filter,sz in enumerate(cfg['filter_sizes']):
        conv = Conv1D(filters=cfg['num_filters'][ ith_filter ], kernel_size=sz, padding="same",
                             activation="relu", strides=1, name=str(ith_filter) + "_thfilter")(emb_lookup)
        conv_blocks.append(GlobalMaxPooling1D()(conv))

    concat = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    concat_drop = Dropout(cfg['dropout'])(concat)

    # different dense layer per tasks
    FC_models = []
    for i in range(len(num_classes)):
        dense = Dense(num_classes[i], name= "Dense"+str(i))( concat_drop )
        FC_models.append(dense)

    # the multitsk model
    model = Model(inputs=model_input, outputs = FC_models)

    return model

# y_true: teacher temperature softmax outputs
# y_pred: student raw logits (need to be transformed and divided by temp)
def kd_loss(y_true,y_pred,temp):
  # for teacher/student softmax comparisons
  y_pred_tsm = K.softmax(y_pred/temp)

  # return error
  return logloss(y_true,y_pred_tsm,from_logits=False)

def main():
  print('\n************************************************************************************', end='\n\n')
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('tech_dir',     type=str, help='Where is the teacher data located?')
  parser.add_argument('dump_dir',     type=str, help='Where are we dumping the output?')
  parser.add_argument('config',       type=int, help='What model config are we using?')

  # Parse all the arguments & set random seed
  args = parser.parse_args()
  seed = int(RANK)
  print('Seed:', seed, end='\n\n')
  np.random.seed(seed)

  # check that dump directory exists
  if not os.path.isdir(args.tech_dir):
    print('TEACHER DIRECTORY DOES NOT EXIST')
    exit(-1)
  # check that dump directory exists
  if not os.path.isdir(args.dump_dir):
    print('DUMP DIRECTORY DOES NOT EXIST')
    exit(-1)

  # Step 1: Get experiment configurations
  config = GetModelConfig(args.config)
  print('run parameters:', config, end='\n\n', flush= True)
  global TEMP
  TEMP = config['temp'][seed]
  print('TEMP:', TEMP, flush= True)

  # Step 2: Create training/testing data for models
  X, XV, XT, Y, YV, YT = loadAllTasks(print_shapes = False)
  del Y,YV
  y,yv = LoadY(args.tech_dir, TEMP)
  print('DATA LOADED AND READY TO GO\n', flush= True)

  # Step 3: Create the studen mtcnn model
  mtcnn = CreateMTCnn(CLASS, max(np.max(X),np.max(XV)) + 1,config)
  print('MODEL CREATED\n', flush= True)

  # create validation data dictionary
  val_dict = {}
  for i in range(len(CLASS)):
    layer = 'Dense' + str(i)
    val_dict[layer] = yv[i]
  print('VALIDATION DICTIONARY CREATED', flush= True)

  # create set of losses
  losses = {}
  for i in range(len(CLASS)):
    l = lambda y_true, y_pred: kd_loss(y_true,y_pred,TEMP)
    l.__name__ = 'kdl'
    losses['Dense'+str(i)] = l
  print('LOSSES CREATED', flush= True)

  mtcnn.compile(loss=losses, optimizer= config['optimizer'], metrics=[ "acc" ])
  print('MODEL COMPILED', flush= True)
  print(mtcnn.summary(), flush= True)


  hist = mtcnn.fit(x= X, y= y,
            batch_size=config['batch_size'],
            epochs=EPOCHS,
            verbose=2,
            validation_data=({'Input': XV}, val_dict),
            callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', restore_best_weights=True)])


  # Step 5: Save everything

  # create directory to dump all data related to model
  fdir = args.dump_dir + 'MTDistilledT-' + str(args.config) + '-' + str(RANK) + '/'
  os.mkdir(fdir)
  micMac = []
  data_path = fdir + "MicMacTest_R" + str(RANK) + ".csv"

  # convert the history.history dict to a pandas DataFrame:
  hist_df = pd.DataFrame(hist.history)
  hist_df.to_csv(path_or_buf= fdir + 'history.csv', index=False)
  print('History Saved!', flush= True)

  # save model
  mtcnn.save(fdir + 'model.h5')
  print('Model Saved!', flush= True)

  # get the softmax values of the our predictions from raw logits
  predT = mtcnn.predict(XT)

  # get the softmax values of the our predictions from raw logits
  for i in range(len(predT)):
    for j in range(len(predT[i])):
      predT[i][j] = softmax(predT[i][j], 1.0)

  for t in range(len(CLASS)):
    preds = np.argmax(predT[t], axis=1)
    micro = f1_score(YT[:,t], preds, average='micro')
    macro = f1_score(YT[:,t], preds, average='macro')
    micMac.append(micro)
    micMac.append(macro)

  data = np.zeros(shape=(1, 10))
  data = np.vstack((data, micMac))
  df0 = pd.DataFrame(data,
                     columns=['Beh_Mic', 'Beh_Mac', 'His_Mic', 'His_Mac', 'Lat_Mic', 'Lat_Mac', 'Site_Mic',
                              'Site_Mac', 'Subs_Mic', 'Subs_Mac'])
  df0.to_csv(data_path)
  print('MIC-MAC SCORES SAVED', flush= True)

  # save model output
  print('Saving Training Softmax Output', flush= True)
  for i in range(len(predT)):
    print('task:',str(i))
    print('--Number of data points: ', len(predT[i]), flush= True)
    print('--Size of each data point', len(predT[i][0]), flush= True)

    fname = fdir + 'testing-task-' + str(i) + '.npy'
    np.save(fname, predT[i])
  print()

if __name__ == '__main__':
  main()