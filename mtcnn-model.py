'''
Created by: Jose Guadalupe Hernandez
Email: jgh9094@gmail.com

Python file will constuct an Multi-Task CNN for the 5 tasks:
behavior, histology, laterality, site, subsite.
'''

# general python imports
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os
import pandas as pd

# keras python inputs
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout
from keras.regularizers import l2
from keras.layers import GlobalMaxPooling1D, Convolution1D
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.layers.merge import Concatenate
from keras import initializers
from keras.losses import CategoricalCrossentropy
from sklearn.metrics import f1_score

from loaddata6reg import loadAllTasks
from mpi4py import MPI


# global variables
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.size #Node count. size-1 = max rank.
# EXPECTED CLASSES FOR EACH TASK, MUST UPDATE
CLASS =  [4,639,7,70,326]
EPOCHS = 100

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
    }

  else:
    print('MODEL CONFIGURATION DOES NOT EXIST', flush= True)
    exit(-1)

# softmax output transformer
def softmax(x):
    ex = np.exp(x)
    tot = np.sum(ex)
    return ex / tot

# transform data and return number of classes
def TransformData(rawX, rawXV, rawXT, rawY, rawYV, rawYT):
  # raw data descriptions
  print('RAW DATA DIMS', flush= True)
  print('rawX dim: ', rawX.shape, flush= True)
  print('rawY dim: ', rawY.shape, flush= True)
  print('rawXV dim: ', rawXV.shape, flush= True)
  print('rawYV dim: ', rawYV.shape, flush= True)
  print('rawXT dim: ', rawXT.shape, flush= True)
  print('rawYT dim: ', rawYT.shape , end='\n\n', flush= True)

  # make sure number of tasks between data sets is consistent
  if rawY.shape[1] != rawYT.shape[1] or rawYT.shape[1] != rawYV.shape[1]:
    print('NUMBER OF TASKS NOT THE SAME BETWEEN DATA SETS', flush= True)
    exit(-1)

  # create array for each task output
  y = [[] for i in range(rawY.shape[1])]
  yv = [[] for i in range(rawY.shape[1])]
  yt = [[] for i in range(rawY.shape[1])]

  # load data into appropiate list
  for t in range(rawY.shape[1]):
    y[t] = rawY[:,t]
    yv[t] = rawYV[:,t]
    yt[t] = rawYT[:,t]

  # make to catagorical data and pack up
  Y,YV,YT = [],[],[]
  for i in range(len(y)):
    Y.append(to_categorical(y[i], num_classes=CLASS[i]))
  for i in range(len(yv)):
    YV.append(to_categorical(yv[i], num_classes=CLASS[i]))
  for i in range(len(yt)):
    YT.append(to_categorical(yt[i], num_classes=CLASS[i]))

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

  print('Testing Output Data', flush= True)
  i = 0
  for y in YT:
    print('task', i, flush= True)
    print('--cases:', len(y), flush= True)
    print('--classes:',len(y[0]), flush= True)
    i += 1
  print()

  # number of classes per task
  classes = []
  for y in Y:
    classes.append(len(y[0]))

  return np.array(rawX),np.array(rawXV),np.array(rawXT),Y,YV,YT,classes

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
        conv = Convolution1D(filters=cfg['num_filters'][ ith_filter ], kernel_size=sz, padding="same",
                             activation="relu", strides=1, name=str(ith_filter) + "_thfilter")(emb_lookup)
        conv_blocks.append(GlobalMaxPooling1D()(conv))

    concat = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    concat_drop = Dropout(cfg['dropout'])(concat)

    # different dense layer per tasks
    FC_models = []
    for i in range(len(num_classes)):
        outlayer = Dense(num_classes[i], name= "Dense"+str(i))( concat_drop )
        FC_models.append(outlayer)

    # the multitsk model
    model = Model(inputs=model_input, outputs = FC_models)
    model.compile( loss= CategoricalCrossentropy(from_logits=True), optimizer= cfg['optimizer'], metrics=[ "acc" ] )

    return model

def main():
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('dump_dir',     type=str, help='Where are we dumping the output?')
  parser.add_argument('config',       type=int, help='What model config are we using?')
  parser.add_argument('prop',         type=float, help='proportion of testcases being used')

  # parse all the argument
  args = parser.parse_args()

  # set the seed to the rank
  seed = int(RANK)
  print('Seed:', seed)
  np.random.seed(seed)

  # Step 1: Get experiment configurations
  print('Config Using:', args.config)
  config = GetModelConfig(args.config)
  print('run parameters:', config, end='\n\n', flush= True)

  # save the dump directory
  dump_dir = args.dump_dir

  # check that dump directory exists
  # we expect the dump dir to be made already
  print('DUMP Directory:', dump_dir)
  if not os.path.isdir(dump_dir):
    print('DUMP DIRECTORY DOES NOT EXIST', flush= True)
    exit(-1)


  # Step 2: Create training/testing data for models
  X, XV, XT, Y, YV, YT= loadAllTasks(print_shapes = False)

  # check that prop is between (0,1]
  if args.prop <= 0 or args.prop > 1:
    print('PROP NOT IN CORRECT RANGE')
    exit(-1)

  # Take the proportion of test cases
  print('PROP:', args.prop)
  propX = int(args.prop * len(X))
  propXV = int(args.prop * len(XV))
  propXT = int(args.prop * len(XT))
  propY = int(args.prop * len(Y))
  propYV = int(args.prop * len(YV))
  propYT = int(args.prop * len(YT))

  # subset the data set
  X = X[0:propX]
  XV = XV[0:propXV]
  XT = XT[0:propXT]
  Y = Y[0:propY]
  YV = YV[0:propYV]
  YT = YT[0:propYT]

  X, XV, XT, Y, YV, YT, classes = TransformData(X, XV, XT, Y, YV, YT)

  # Step 3: Create the mtcnn model
  mtcnn = CreateMTCnn(classes, max(np.max(X),np.max(XV)) + 1,config)
  print( mtcnn.summary() )

  # Step 4: Train mtcnn model

  # create validation data dictionary
  val_dict = {}
  for i in range(len(YV)):
    layer = 'Dense' + str(i)
    val_dict[layer] = YV[i]

  hist = mtcnn.fit(x= X, y= Y, batch_size= config['batch_size'],
          epochs= EPOCHS, verbose= 2, validation_data= ({'Input': XV}, val_dict),
          callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True)]
          )

  # create directory to dump all data related to model
  fdir = dump_dir + 'MTModel-' + str(args.config) + "_Rank-" + str(RANK) +'/'
  if not os.path.exists(fdir):
    os.makedirs(fdir)

  # save predictions from all data inputs

  pred = mtcnn.predict(X)
  predV = mtcnn.predict(XV)
  predT = mtcnn.predict(XT)

  print('Saving Training Softmax Output', flush= True)
  for i in range(len(pred)):
    print('task:',str(i))
    print('--Number of data points: ', len(pred[i]), flush= True)
    print('--Size of each data point', len(pred[i][0]), flush= True)

    fname = fdir + 'training-task-' + str(i) + '.npy'
    np.save(fname, pred[i])
  print()

  print('Saving Validation Softmax Output', flush= True)
  for i in range(len(predV)):
    print('task:',str(i), flush= True)
    print('--Number of data points: ', len(predV[i]), flush= True)
    print('--Size of each data point', len(predV[i][0]), flush= True)

    fname = fdir + 'validating-task-' + str(i) + '.npy'
    np.save(fname, predV[i])
  print()

  print('Saving Testing Softmax Output', flush= True)
  for i in range(len(predT)):
    print('task:',str(i))
    print('--Number of data points: ', len(predT[i]), flush= True)
    print('--Size of each data point', len(predT[i][0]), flush= True)

    fname = fdir + 'testing-task-' + str(i) + '.npy'
    np.save(fname, predT[i])
  print()

  #predT has this shape: [numTasks, numSamples, numLabelsInTask]
  '''
  Save final micro/macro:
  '''
  micMac = []

  data_path = fdir + "MicMacTest_R" + str(RANK) + ".csv"
  X, XV, XT, Y, YV, YT= loadAllTasks(print_shapes = False)

  # subset the data set
  X = X[0:propX]
  XV = XV[0:propXV]
  XT = XT[0:propXT]
  Y = Y[0:propY]
  YV = YV[0:propYV]
  YT = YT[0:propYT]

  # get the softmax values of the our predictions from raw logits
  for i in range(len(predT)):
    for j in range(len(predT[i])):
      predT[i][j] = softmax(predT[i][j])

  for t in range(5):
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

  # convert the history.history dict to a pandas DataFrame:
  hist_df = pd.DataFrame(hist.history)
  hist_df.to_csv(path_or_buf= fdir + 'history.csv', index=False)
  print('History Saved!', flush= True)

  # save model
  mtcnn.save(fdir + 'model.h5')
  print('Model Saved!', flush= True)


if __name__ == '__main__':
  main()
