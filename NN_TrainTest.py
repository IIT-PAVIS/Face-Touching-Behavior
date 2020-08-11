import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras import regularizers
import scipy.io as sio
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from time import time
from glob import glob
import os
import matplotlib.pyplot as plt
from sklearn import metrics

###########Parameters
NB_EPOCH = 100
BATCH_SIZE = 2048
num_classes = 2 # Total Number of classes
VALIDATION_SPLIT=0.2 #Validation split that can be used for parameter setting
##########################
seed = 1990
np.random.seed(1990) #for random initialization

basePathFeature94 = './features/' #refers to features called 'Face parts and hand key-points' in Beyan et al. ICMI 2020
basePathFeature11 = './feature11D/' #refers to features called 'Face and hands bounding boxes' in Beyan et al. ICMI 2020
basePathLabel   = './T1_2_3_4_5/' #refers to cross validation splits
subdir = glob(basePathLabel+'*/')
testsets = [1,2,3,4,5]
for folds in range(len(testsets)):
    print('testsets = ',testsets[folds] )
    train_x = np.empty([0, 105],dtype=np.float32)
    train_y = np.empty([0, 1])
    test_x  = np.empty([0, 105],dtype=np.float32)
    test_y = np.empty([0, 1])
    for subfoldr in subdir:
        filelist = [file for file in os.listdir(subfoldr) if file.endswith('.mat')]
        for file in filelist:
            label = sio.loadmat(subfoldr+file)
            data94D = sio.loadmat(basePathFeature94 + file)
            data11D = sio.loadmat(basePathFeature11 + file)
            tempD = np.hstack([data94D['face_hand'], data11D['face_hand']])
            lbl = label['final']
            if(testsets[folds]==int(subfoldr[-2:][0])):
                test_x = np.vstack([test_x, tempD])# = np.stack(test_x, data['face_hand'])
                test_y = np.vstack([test_y, lbl])#= [test_y, label['final']]
            else:

                train_x = np.vstack([train_x, tempD])  # = np.stack(test_x, data['face_hand'])
                train_y = np.vstack([train_y, lbl])

    ################preapring training data ##############
    ######normalization of training and test data
    maxColmn= np.amax(train_x, axis=0)
    train_x = train_x / maxColmn[None,:] #feature normalization
    test_x  = test_x / maxColmn[None,:]
    ########################################
    input_shape = train_x.shape[1]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train_y, num_classes)

    y_test = keras.utils.to_categorical(test_y, num_classes)
    #y_test = y_test[:, 1:]
    # model creation
    total_Sample = len(train_y)
    epoches = 100#[10,20,50,100] model.add(Dense(64, input_dim=64, kernel_regularizer=regularizers.l2(0.01)))
    #################defining a sequential model############
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.30))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.30))
    model.add(Dense(512, activation='relu'))# metrics=[tf.keras.metrics.AUC()]
    model.add(BatchNormalization())
    model.add(Dropout(0.30))
    model.add(Dense(num_classes, activation='softmax')) #name="binary_crossentropy" categorical_crossentropy
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=1e-3))#,class_weight=weights)
    #model.summary()
    history = model.fit(train_x, y_train, verbose=0, batch_size=BATCH_SIZE, epochs=epoches,validation_data=(test_x, y_test))
    score = model.evaluate(test_x, y_test, verbose=0)
    ypred = model.predict(test_x)
    ##############################
    # tempFc =
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    #print('Probs',probs)
    lablP = np.argmax(ypred, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    print(metrics.confusion_matrix(true_labels, lablP))
    TP = TP + np.sum(np.logical_and(lablP == 1, true_labels == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = TN + np.sum(np.logical_and(lablP == 0, true_labels == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = FP + np.sum(np.logical_and(lablP == 1, true_labels == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = FN + np.sum(np.logical_and(lablP == 0, true_labels == 1))

    print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP, FP, TN, FN))
    PRCN = np.float(TP)/np.float(TP+FP+0.000001)
    RCALL= np.float(TP) / np.float(TP + FN + 0.000001)
    F1Score = 2*PRCN*RCALL/(PRCN+RCALL+0.0000001)
    print('Percision: %f, Recall: %f, Fscore: %f'%(PRCN, RCALL, F1Score))