'''
Created by: Jose Guadalupe Hernandez
Email: jgh9094@gmail.com

This file will distill the knowledge from teachers into a Multi-task CNN.
Using two loss functions: hard label loss and distillation loss.
'''

# general python imports
import numpy as np
import argparse
import os
import pandas as pd

# keras python inputs
from keras.models import Model
from keras.layers import Input,Embedding,Dropout,Dense,GlobalMaxPooling1D,Conv1D,Lambda,Activation,concatenate
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy as logloss
from keras import backend as K
from keras import initializers
from keras.metrics import categorical_accuracy
from keras.layers.merge import Concatenate
from sklearn.metrics import f1_score

# summit specific imports
from loaddata6reg import loadAllTasks
from mpi4py import MPI

# global variables
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.size
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
      'batch_size': 5,
      'dropout': 0.5,
      'optimizer': 'adam',
      'wv_len': 300,
      'emb_l2': 0.001,
      'in_seq_len': 1500,
      'filter_sizes': [3,4,5],
      'num_filters': [300,300,300],
      'alpha': 0.07,
      'temp': [0.1,0.2,0.5,0.7,0.9,1.0,1.1,1.2,1.5,1.7,1.9,2.0]
    }
  else:
    print('MODEL CONFIGURATION DOES NOT EXIST')
    exit(-1)

# first 1/2 are hard labels, second 1/2 are softmax outputs
# alpha: constant for hard label error (should be small according to lit review)
def kd_loss(y_true,y_pred,alpha,temp,split):
  # ground truth and teacher softmax
  y_true_hl, y_true_sl = y_true[: , :split], y_true[: , split:]
  # student softmax and logits/temp
  s_softmax, s_logits = y_pred[: , :split], y_pred[: , split:]
  # student raw logtis tranformed/temp: distallation loss
  s_logits_sl = K.softmax(s_logits/temp)

  # hard loss
  hl = alpha * logloss(y_true_hl,s_softmax,from_logits=False)
  # distillation loss
  dl = (1.0 - alpha) * logloss(y_true_sl,s_logits_sl,from_logits=False)

  return hl + dl

# softmax/temperture output transformer
def softmax(x,t):
    ex = np.exp(x/t)
    tot = np.sum(ex)
    return np.array(ex / tot)

# concatenate the data
# 1st half: hard lables
# 2nd half: teacher temperature softmax
def ConcatData(y,yv,teach, temp):
  print('CONCAT DATA:', flush= True)
  Y,YV = [],[]
  # iterate through the number of classes in the training data
  for i in range(len(CLASS)):
    print(str(i))
    Y.append([])

    # get training dir
    yt = np.load(teach + 'training-task-' + str(i) + '.npy')
    # make sure same lengths
    if yt.shape[0] != y[i].shape[0] or yt.shape[1] != y[i].shape[1]:
      print('NOT MATHCING DIMENSIONS: TRAINING')
      exit(-1)
    # concatenate + transform the teacher data the output data
    for j in range(yt.shape[0]):
      Y[i].append(np.concatenate((y[i][j], softmax(yt[j], temp))))
    # make a numpy array
    Y[i] = np.array(Y[i])

    # get validation dir
    YV.append([])
    yvt = np.load(teach + 'validating-task-' + str(i) + '.npy')
    # make sure same lengths
    if yvt.shape[0] != yv[i].shape[0] or yvt.shape[1] != yv[i].shape[1]:
      print('NOT MATHCING DIMENSIONS: VALIDATING')
      exit(-1)
    # concatenate + transform the teacher data the output data
    for j in range(yvt.shape[0]):
      YV[i].append(np.concatenate((y[i][j], softmax(yvt[j], temp))))
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

# transform y data to_categorical
# [0,1,0,1,...]
def Transform(rawY,rawYV):
  print('TRANSFORM', flush= True)
  # create array for each task output
  # create array for each task output
  y = [[] for i in range(rawY.shape[1])]
  yv = [[] for i in range(rawY.shape[1])]

  # load data into appropiate list
  for t in range(rawY.shape[1]):
    y[t] = rawY[:,t]
    yv[t] = rawYV[:,t]

  # make to catagorical data and pack up
  Y,YV = [],[]
  for i in range(len(y)):
    Y.append(to_categorical(y[i], num_classes=CLASS[i]))
  for i in range(len(yv)):
    YV.append(to_categorical(yv[i], num_classes=CLASS[i]))

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
def CreateMTCnn(num_classes,vocab_size,cfg):
    # define network layers ----------------------------------------------------
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
        dense = Dense(num_classes[i], name= "Dense"+str(i), )( concat_drop )
        act = Activation('softmax', name= "Active"+str(i))(dense)
        FC_models.append(act)

    # the multitsk model
    model = Model(inputs=model_input, outputs = FC_models)
    model.compile( loss= "categorical_crossentropy", optimizer= cfg['optimizer'], metrics=[ "acc" ] )

    return model

def main():
  print('************************************************************************************', flush= True)
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('tech_dir',     type=str, help='Where is the teacher data located?')
  parser.add_argument('dump_dir',     type=str, help='Where are we dumping the output?')
  parser.add_argument('config',       type=int, help='What model config are we using?')

  # Parse all the arguments & set random seed
  args = parser.parse_args()
  # RANK = 0
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
  print('run parameters:', config, flush= True)
  global TEMP
  TEMP = config['temp'][seed]
  print('TEMP:', TEMP, flush= True)

  # Step 2: Create training/testing data for models
  X, XV, XT, Y, YV, YT = loadAllTasks(print_shapes = False)
  Y,YV = Transform(Y,YV)
  Y,YV = ConcatData(Y,YV, args.tech_dir, TEMP)
  print('DATA LOADED AND READY TO GO', flush= True)

  # Step 3: Create the studen mtcnn model
  mtcnn = CreateMTCnn(CLASS, max(np.max(X),np.max(XV)) + 1,config)
  print('MODEL CREATED\n', flush= True)

  #Step 4: Create knowledge distilled student topology

  # remove the last activation layers
  for i in range(len(CLASS)):
    mtcnn.layers.pop()

  # add new distlled layers
  new_out = []
  for i in range(len(CLASS)):
    logits = mtcnn.get_layer('Dense'+str(i)).output
    probs = Activation('softmax', name='Logits'+str(i))(logits)
    # softed probabilities at raised temperature
    logits_T = Lambda(lambda x: x, name='lambda'+str(i))(logits)
    # probs_T = Activation('softmax', name='TLogits'+str(i))(logits_T)
    # output layer: softmax from dense & raw logits
    output = concatenate([probs, logits_T], name="Active"+str(i))
    new_out.append(output)

  # mtcnn distillation model ready to go!
  mtcnn = Model(mtcnn.input, new_out)
  mtcnn.summary()
  print('MODEL READJUSTED FOR DISTILLATION\n')

  # student softmax(raw_logits) and hard labels
  def acc(y_true, y_pred, split):
    y_true = y_true[:, :split]
    y_pred = y_pred[:, split:]
    y_pred = K.softmax(y_pred)
    return categorical_accuracy(y_true, y_pred)

  # student softmax(raw_logits) and hard labels
  def categorical_crossentropy(y_true, y_pred, split):
    y_true = y_true[:, :split]
    y_pred = y_pred[:, split:]
    return logloss(y_true, y_pred, from_logits=True)

  # student softmax(raw_logits) and hard labels
  def soft_logloss(y_true, y_pred, split,temp):
    y_true = y_true[:, split:]
    y_pred = y_pred[:, split:]
    y_soft = y_pred/temp
    return logloss(y_true, y_soft, from_logits=True)

  # create loss dictionary for each task
  losses = {}
  # task 0
  l0 = lambda y_true, y_pred: kd_loss(y_true,y_pred,config['alpha'], TEMP,CLASS[0])
  l0.__name__ = 'kdl'+str(0)
  losses['Active'+str(0)] = l0
  # task 1
  l1 = lambda y_true, y_pred: kd_loss(y_true,y_pred,config['alpha'], TEMP,CLASS[1])
  l1.__name__ = 'kdl'+str(1)
  losses['Active'+str(1)] = l1
  # task 2
  l2 = lambda y_true, y_pred: kd_loss(y_true,y_pred,config['alpha'], TEMP,CLASS[2])
  l2.__name__ = 'kdl'+str(2)
  losses['Active'+str(2)] = l2
  # task 3
  l3 = lambda y_true, y_pred: kd_loss(y_true,y_pred,config['alpha'], TEMP,CLASS[3])
  l3.__name__ = 'kdl'+str(3)
  losses['Active'+str(3)] = l3

  # create metric dictionary per task
  metrics = {}
  # task 0
  metrics['Active'+str(0)] = []
  l01 = lambda y_true, y_pred: acc(y_true,y_pred,CLASS[0])
  l01.__name__ = 'acc'+str(0)
  metrics['Active'+str(0)].append(l01)
  l02 = lambda y_true, y_pred: categorical_crossentropy(y_true,y_pred,CLASS[0])
  l02.__name__ = 'cc'+str(0)
  metrics['Active'+str(0)].append(l02)
  l03 = lambda y_true, y_pred: soft_logloss(y_true,y_pred,CLASS[0],TEMP)
  l03.__name__ = 'sl'+str(0)
  metrics['Active'+str(0)].append(l03)
  # task 1
  metrics['Active'+str(1)] = []
  l11 = lambda y_true, y_pred: acc(y_true,y_pred,CLASS[1])
  l11.__name__ = 'acc'+str(1)
  metrics['Active'+str(1)].append(l11)
  l12 = lambda y_true, y_pred: categorical_crossentropy(y_true,y_pred,CLASS[1])
  l12.__name__ = 'cc'+str(1)
  metrics['Active'+str(1)].append(l12)
  l13 = lambda y_true, y_pred: soft_logloss(y_true,y_pred,CLASS[1],TEMP)
  l13.__name__ = 'sl'+str(1)
  metrics['Active'+str(1)].append(l13)
  # task 2
  metrics['Active'+str(2)] = []
  l21 = lambda y_true, y_pred: acc(y_true,y_pred,CLASS[2])
  l21.__name__ = 'acc'+str(2)
  metrics['Active'+str(2)].append(l21)
  l22 = lambda y_true, y_pred: categorical_crossentropy(y_true,y_pred,CLASS[2])
  l22.__name__ = 'cc'+str(2)
  metrics['Active'+str(2)].append(l22)
  l23 = lambda y_true, y_pred: soft_logloss(y_true,y_pred,CLASS[2],TEMP)
  l23.__name__ = 'sl'+str(2)
  metrics['Active'+str(2)].append(l23)
  # task 3
  metrics['Active'+str(3)] = []
  l31 = lambda y_true, y_pred: acc(y_true,y_pred,CLASS[3])
  l31.__name__ = 'acc'+str(3)
  metrics['Active'+str(3)].append(l31)
  l32 = lambda y_true, y_pred: categorical_crossentropy(y_true,y_pred,CLASS[3])
  l32.__name__ = 'cc'+str(3)
  metrics['Active'+str(3)].append(l32)
  l33 = lambda y_true, y_pred: soft_logloss(y_true,y_pred,CLASS[3],TEMP)
  l33.__name__ = 'sl'+str(3)
  metrics['Active'+str(3)].append(l33)

  print('METRICS CREATED', flush= True)

  # create validation data dictionary
  val_dict = {}
  for i in range(len(CLASS)):
    layer = 'Active' + str(i)
    val_dict[layer] = YV[i]
  print('VAL-DICT CREATED', flush= True)

  mtcnn.compile(optimizer='adam', loss=losses, metrics=metrics)
  print('MODEL COMPILED FOR DISTILLATION\n', flush= True)

  hist = mtcnn.fit(X, Y,
            batch_size=config['batch_size'],
            epochs=EPOCHS,
            verbose=2,
            validation_data=({'Input': XV}, val_dict),
            callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', restore_best_weights=True)])

  # Step 5: Save everything

  # create directory to dump all data related to model
  fdir = args.dump_dir + 'MTDistilled2L-' + str(args.config) + '-' + str(RANK) + '/'
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

  # use only the first half of the output vector: those are predictions
  pred = [[] for x in range(len(CLASS))]

  # get the softmax values of the our predictions from raw logits
  for i in range(len(predT)):
    for j in range(len(predT[i])):
      pred[i].append(softmax(predT[i][j][CLASS[i]:],1.0))

  pred = [np.array(x) for x in pred]

  for t in range(len(CLASS)):
    preds = np.argmax(pred[t], axis=1)
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