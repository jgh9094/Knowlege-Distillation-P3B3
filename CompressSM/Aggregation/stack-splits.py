'''
Created by: Jose Guadalupe Hernandez
Email: jgh9094@gmail.com

Will combine all the partitions that were split up.
'''

# general python imports
import numpy as np
import argparse
import pickle as pk
import psutil
import os

# Get total number of files there should be for each data type
def GetTotal(c):
  # training data
  if c == 0:
    return 160
  # testing data
  elif c == 1:
    return 20
  # validating data
  elif c == 2:
    return 18
  else:
    print('UNKNOWN TOTAL')
    exit(-1)

# Get file naming for each data type
def GetDataType(c,t):
  # training data
  if c == 0:
    return 'training-task-' + str(t) + '-rank-'
  # testing data
  elif c == 1:
    return 'testing-task-' + str(t) + '-rank-'
  # validating data
  elif c == 2:
    return 'validating-task-' + str(t) + '-rank-'
  else:
    print('UNKNOWN TOTAL')
    exit(-1)

# will look at all directories in data dir and sample a set of them
def GetDataFiles(dir,fn,n):
  dirs = [dir + fn + str(i) + '.npy' for i in range(n)]

  print('DIRS EXPLORING:', flush= True)
  for d in dirs:
    print(d, flush= True)
  print(flush= True)

  return dirs

# will return string associated with data type
def GetData(c):
  # training data
  if c == 0:
    return 'training'
  # testing data
  elif c == 1:
    return 'testing'
  # validating data
  elif c == 2:
    return 'validating'
  else:
    print('UNKNOWN TOTAL')
    exit(-1)

# will look through all dirs and average out their data (testing, training, validate)
def ConcatData(files,dump,data,task):
  print('PROCESSING FILES', flush=True)

  # will hold all data to stack
  hold = []

  for f in files:
    print('processing:', f, flush= True)
    X = np.load(file=f)
    hold.append(X)

  mat = np.concatenate(hold)
  print('mat.shape', mat.shape)

  np.save(dump + data + '-task-' + str(task) +'.npy', mat)
  print('finished saving:', dump + data + '-task-' + str(task) +'.npy', flush= True)
  print(flush= True)


def main():
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('data_dir',     type=str,      help='Where are we dumping the output?')
  parser.add_argument('dump_dir',     type=str,      help='Where are we dumping the output?')
  parser.add_argument('task',         type=int,      help='What task are we looking for')
  parser.add_argument('data_type',    type=int,      help='0: training, 1: testing, 2: validating')

  # parse all the argument
  args = parser.parse_args()

  # what are the inputs
  print('data_dir:', args.data_dir, flush= True)
  print('dump_dir:', args.dump_dir, flush= True)
  print('task:', args.task, flush= True)
  print('data_type:', args.data_type, flush= True)

  # Step 1: Get data directories we are exploring
  fn = GetDataType(args.data_type, args.task)
  n = GetTotal(args.data_type)
  files = GetDataFiles(args.data_dir.strip(),fn,n)

  # Step 2: Concatenate the data
  data = GetData(args.data_type)
  ConcatData(files, args.dump_dir, data,args.task)


if __name__ == '__main__':
  main()
