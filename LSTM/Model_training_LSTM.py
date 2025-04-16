import csv
import os
import sys
import numpy as np
import pandas as pd
import operator

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import normalize 
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import TensorBoard
from timeit import default_timer as timer
import time

dataPath =  r"C:\Users\Abbas\Desktop\ids-project"
resultPath =  r"C:\Users\Abbas\Desktop\ids-project\results"
if not os.path.exists(resultPath):
    print('result path {} created.'.format(resultPath))
    os.mkdir(resultPath)
    
dep_var = 'Label'
model_name = "init"

cat_names = ['Dst Port', 'Protocol']
cont_names = ['Timestamp', 'Flow Duration', 'Tot Fwd Pkts',
              'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
              'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
              'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
              'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
              'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
              'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
              'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
              'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
              'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
              'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
              'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
              'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
              'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
              'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
              'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
              'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
              'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
              'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
              'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
              'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']

def loadData(fileName):
    dataFile = os.path.join(dataPath, fileName)
    pickleDump = '{}.pickle'.format(dataFile)
    if os.path.exists(pickleDump):
        df = pd.read_pickle(pickleDump)
    else:
        df = pd.read_csv(dataFile)
        df = df.dropna()
        df = shuffle(df)
        df.to_pickle(pickleDump)
    return df

def baseline_model(inputDim=-1, out_shape=(-1,)):
    global model_name
    model = Sequential()
    if inputDim > 0 and out_shape[1] > 0:
       # Changed Replace Dense with LSTM layer and adjusted input shape 
        time_steps = 1 # Single timestep for non-sequential data 
        model.add(LSTM(100, input_shape=(time_steps, inputDim))) #3D input format

        # Changed Removed second Dense Layer for LSTM simplicity
        model.add(Dense(out_shape[1], activation='softmax'))# Output layer

        if out_shape[1] > 2:
            print('Categorical Cross-Entropy Loss Function')
            model_name += "_categorical"
            model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        else:
            model_name += "_binary"
            print('Binary Cross-Entropy Loss Function')
            model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    return model

def load_model_csv(_model_name):
    #Change to your own path
    model = load_model('results/models/02-14-2018/02-14-2018.csv_1743864950_categorical.h5'.format(_model_name))
    return model

def experiment(dataFile, optimizer='adam', epochs=10, batch_size=10):
    
    #Creating data for analysis
    time_gen = int(time.time())
    global model_name
    model_name = f"{dataFile}_{time_gen}"
    #$ tensorboard --logdir=logs/
    tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
    
    seed = 7
    np.random.seed(seed)
    cvscores = []
    print('optimizer: {} epochs: {} batch_size: {}'.format(
        optimizer, epochs, batch_size))
    
    data = loadData(dataFile)
    data_y = data.pop('Label')
    
    #transform named labels into numerical values
    encoder = LabelEncoder()
    encoder.fit(data_y)
    data_y = encoder.transform(data_y)
    dummy_y = to_categorical(data_y)
    data_x = normalize(data.values)
    
    #define 5-fold cross validation test harness
    inputDim = len(data_x[0])
    print('inputdim = ', inputDim)
    
  
    #Separate out data
    #X_train, X_test, y_train, y_test = train_test_split(data_x, dummy_y, test_size=0.2)
    num=0
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
    start = timer()
    for train_index, test_index in sss.split(X=np.zeros(data_x.shape[0]), y=dummy_y):
        X_train, X_test = data_x[train_index], data_x[test_index]
        y_train, y_test = dummy_y[train_index], dummy_y[test_index]

         # Convert from 2D to 3D (samples, timesteps, features)
        timesteps = 1  # Single timestep since we're not using sequences
        X_train = X_train.reshape((X_train.shape[0], timesteps, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], timesteps, X_test.shape[1]))

        #create model
        model = baseline_model(inputDim, y_train.shape)
    
        #train
        print("Training " + dataFile + " on split " + str(num))
        model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=32, verbose=2, callbacks=[tensorboard], validation_data=(X_test, y_test))

        #save model
        model.save(f"{resultPath}/models/{model_name}_LSTM.h5")

        num+=1

    elapsed = timer() - start

    X_test = X_test.reshape((X_test.shape[0], timesteps, X_test.shape[2]))
    scores = model.evaluate(X_test, y_test, verbose=1)

    print(model.metrics_names)
    acc, loss = scores[1]*100, scores[0]*100
    print('Baseline: accuracy: {:.2f}%: loss: {:.2f}'.format(acc, loss))

   # Extract file name without extension for directory naming
    file_name_without_ext = os.path.splitext(dataFile)[0]
    
    # Create results directory with input file name (without extension)
    specific_result_path = os.path.join(resultPath, file_name_without_ext)
    os.makedirs(specific_result_path, exist_ok=True)
    
    # Save results to text file
    result_file_path = os.path.join(specific_result_path, f"{file_name_without_ext}_results.txt")
    try:
        with open(result_file_path, 'a') as fout:
            fout.write(f'{model_name} results...\n')
            fout.write(f'\taccuracy: {acc:.2f}% loss: {loss:.2f}\n')
            fout.write(f'\telapsed time: {elapsed:.2f} sec\n')
        print(f"Results saved to {result_file_path}")
    except IOError as e:
        print(f"Model Saved successfully")
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python(3) keras-tensorflow.py inputFile.csv (do not include full path to file)")
    else:
        experiment(sys.argv[1])
        
