'''
Created by: Kevin De Angeli
Email: Kevindeangeli@utk.edu
Date: 7/2/21

Y: behavior  histology  laterality site subsite
# classes:  [4, 639,   7,  70, 326]


'''
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn import preprocessing
import pickle
DataPath = "/gpfs/alpine/proj-shared/med107/NCI_Data/tokenized20210406/"
newDataPath = "//gpfs/alpine/med107/proj-shared/kevindeangeli/EnsembleDestilation/preprocessedData/"
HardLabelsPath = "/gpfs/alpine/world-shared/med106/yoonh/storageFolder/HardLabels/"

taskDic = {"Behavior": 0, "Histology": 1, "Laterality": 2, "Site": 3, "Subsite": 4}


def PreprocessX():

    tr_x=pd.read_csv(DataPath+"X_train.csv")
    tr_x =tr_x["concatenatedTextFields"]
    print(len(tr_x))

    va_x =pd.read_csv(DataPath+"X_val.csv")
    va_x =va_x["concatenatedTextFields"]
    print(len(va_x))


    te_x =pd.read_csv(DataPath+"X_test.csv")
    te_x =te_x["concatenatedTextFields"]
    print(len(te_x))



    Data = []
    for document in tr_x:
        dataSplit = document[1:-1].split(', ') #ignore "[" and "]"
        Data.append(list(map(int, dataSplit)))
    train_x = pad_sequences(Data, maxlen= 1500, padding= 'pre', truncating= 'pre' )
    train_x = np.array(train_x)
    with open(newDataPath + 'train_x.pickle', 'wb') as handle:
        pickle.dump(train_x, handle, protocol=pickle.HIGHEST_PROTOCOL)

    Data = []
    for document in va_x:
        dataSplit = document[1:-1].split(', ') #ignore "[" and "]"
        Data.append(list(map(int, dataSplit)))
    val_x = pad_sequences(Data, maxlen= 1500, padding= 'pre', truncating= 'pre' )
    val_x = np.array(val_x)
    with open(newDataPath + 'val_x.pickle', 'wb') as handle:
        pickle.dump(val_x, handle, protocol=pickle.HIGHEST_PROTOCOL)

    Data = []
    for document in te_x:
        dataSplit = document[1:-1].split(', ') #ignore "[" and "]"
        Data.append(list(map(int, dataSplit)))
    test_x = pad_sequences(Data, maxlen= 1500, padding= 'pre', truncating= 'pre' )
    test_x = np.array(test_x)
    with open(newDataPath + 'test_x.pickle', 'wb') as handle:
        pickle.dump(test_x, handle, protocol=pickle.HIGHEST_PROTOCOL)



def PreprocessY():
    '''
    Put the data in the format ["behavior", "histology", "laterality", "site", "subsite"]
    Save the enconder.
    '''


    tasks = ["behavior", "histology", "laterality", "site", "subsite"]

    train_size = len(pd.read_csv(DataPath+"Y_train.csv"))
    val_size = len(pd.read_csv(DataPath+"Y_val.csv"))
    test_size = len(pd.read_csv(DataPath+"Y_test.csv"))


    train_y = np.zeros((train_size, len(tasks)))
    val_y = np.zeros((val_size, len(tasks)))
    test_y = np.zeros((test_size, len(tasks)))



    for idx, t in enumerate(tasks):

        tr_y=pd.read_csv(DataPath+"Y_train.csv")
        tr_y =list(tr_y[t])

        va_y =pd.read_csv(DataPath+"Y_val.csv")
        va_y =list(va_y[t])

        te_y =pd.read_csv(DataPath+"Y_test.csv")
        te_y =list(te_y[t])

        Y = np.concatenate((tr_y,va_y,te_y))
        le = preprocessing.LabelEncoder()
        le.fit(Y)

        train = le.transform(tr_y)
        train_y[:,idx]= train

        val = le.transform(va_y)
        val_y[:,idx]= val

        test = le.transform(te_y)
        test_y[:,idx]= test

        #Save Encoder per task
        with open(newDataPath+ t + '_le.pickle', 'wb') as handle:
            pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #save new dataset format:
    with open(newDataPath+ 'train_y.pickle', 'wb') as handle:
        pickle.dump(train_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(newDataPath+ 'val_y.pickle', 'wb') as handle:
        pickle.dump(val_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(newDataPath+ 'test_y.pickle', 'wb') as handle:
        pickle.dump(test_y, handle, protocol=pickle.HIGHEST_PROTOCOL)



def test_le():
    '''
    For debugging purpose
    '''
    le = pickle.load(open(newDataPath+'histology_le.pickle', "rb"))
    tr_y=pd.read_csv(DataPath+"Y_test.csv")
    tr_y =list(tr_y["histology"])
    encoded_labels = le.transform(tr_y)
    print(encoded_labels[0:10])

def loadY():
    train_y = pickle.load(open(HardLabelsPath+'train_y.pickle', "rb"))
    test_y = pickle.load(open(HardLabelsPath+'test_y.pickle', "rb"))
    val_y = pickle.load(open(HardLabelsPath+'val_y.pickle', "rb"))
    #get number of classes
    #print(np.max(np.concatenate((train_y,test_y,val_y)),axis=0)) #[  3. 638.   6.  69. 325.]
    # print(train_y.shape)
    # print(test_y.shape)
    # print(val_y.shape)
    # print(train_y[0:5])
    return train_y, val_y, test_y



def loadX():
    train_x = pickle.load(open(newDataPath+'train_x.pickle', "rb"))
    test_x = pickle.load(open(newDataPath+'test_x.pickle', "rb"))
    val_x = pickle.load(open(newDataPath+'val_x.pickle', "rb"))
    return train_x, val_x, test_x

def loadAllTasks(print_shapes = False):
    train_x, val_x, test_x = loadX()
    train_y, val_y, test_y = loadY()
    if print_shapes is True:
        print("Train X: ", train_x.shape)
        print("Train Y: ", train_y.shape)
        print("Val X: ", val_x.shape)
        print("Val Y: ", val_y.shape)
        print("Test X: ", test_x.shape)
        print("Test Y: ", test_y.shape)
    return train_x, val_x, test_x, train_y, val_y, test_y

def loadSingleTask(task,print_shapes = False):
    train_x, val_x, test_x, train_y, val_y, test_y = loadAllTasks()
    train_y = train_y[:,taskDic[task]]
    test_y = test_y[:, taskDic[task]]
    val_y = val_y[:, taskDic[task]]
    if print_shapes is True:
        print("Train X: ", train_x.shape)
        print("Train Y: ", train_y.shape)
        print("Val X: ", val_x.shape)
        print("Val Y: ", val_y.shape)
        print("Test X: ", test_x.shape)
        print("Test Y: ", test_y.shape)
    return train_x, val_x, test_x, train_y, val_y, test_y







if __name__ == "__main__":
    #puts the data in the format Y = ["behavior", "histology", "laterality", "site", "subsite"]
    #PreprocessY()
    #PreprocessX()
    #loadY()
    train_x, val_x, test_x, train_y, val_y, test_y= loadAllTasks(True)
    #train_x, val_x, test_x, train_y, val_y, test_y= loadSingleTask("Histology",print_shapes = True)
    #print(max(train_y))
    #print(type(max(train_y)))
    X = np.concatenate((train_x,val_x,test_x))
    print("vocab size: ",np.max(X))








