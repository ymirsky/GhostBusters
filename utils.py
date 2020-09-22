import numpy as np
import os
import cv2 as cv

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import AUC
import sklearn.metrics as metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

# Given the paths to real and fake cropped signs (positivly detected by an object classifier), create a training set and test set.
# The data is split chronologically (by filename) to avoid having the same signs in the test set (since we extracted from videos).
# Return: the FILEPATHS to the npy images for the train and test set. We load the images later during training to reduce memory consumption.
def split_data(base_path_r,base_path_f,split=0.8):
    X_r = []
    X_r += [os.path.join(base_path_r,file) for file in os.listdir(base_path_r)]
    Y_r = np.zeros((len(X_r),2))
    Y_r[:,0] = 1 # 0:real, 1:fake
    X_f = []
    X_f += [os.path.join(base_path_f,file) for file in os.listdir(base_path_f)]
    Y_f = np.zeros((len(X_f),2))
    Y_f[:,1] = 1 # 0:real, 1:fake

    # Real : train test
    split_indx = int(len(X_r)*split)
    Xtrain_r = X_r[:split_indx]
    Ytrain_r = Y_r[:split_indx,:]
    Xtest_r = X_r[split_indx:]
    Ytest_r = Y_r[split_indx:,:]

    # Fake : train test
    split_indx = int(len(X_f)*split)
    Xtrain_f = X_f[:split_indx]
    Ytrain_f = Y_f[:split_indx,:]
    Xtest_f = X_f[split_indx:]
    Ytest_f = Y_f[split_indx:,:]

    # Train test
    Xtrain = np.array(Xtrain_r + Xtrain_f)
    Ytrain = np.vstack((Ytrain_r,Ytrain_f))
    Xtest = np.array(Xtest_r + Xtest_f)
    Ytest = np.vstack((Ytest_r,Ytest_f))
    
    return Xtrain,Ytrain,Xtest,Ytest

# Used by the Keras model to dynamically load the images during training and prepair their features
# if dynamic_loading, then images are loaded from disk during training to save memory
class DataLoader(keras.utils.Sequence):
    def __init__(self, X, Y, batch_size=32, shuffle=True,dynamic_loading=False):
        #'Initialization'
        self.batch_size = batch_size
        self.Y = Y
        self.X = X
        self.filenames = [x.split('/')[-1] for x in X] #take the filname, not the full path
        self.shuffle = shuffle
        self.dynamic_loading=dynamic_loading
        if not self.dynamic_loading: #load all data to RAM
            self.load_all()
        self.on_epoch_end()

    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def load_all(self):
        print('Loading files from disk:',self.X[0],'...')
        Xd=[]
        for x in self.X:
            x_img=np.load(x)
            Xd.append(x_img)
        self.X = np.array(Xd)
    
    def __getitem__(self, index):
        #'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        if not self.dynamic_loading:
            return self.X[indexes,], self.Y[indexes,]
        else:
            Xd = []
            for x in self.X[indexes]:
                x_img=np.load(x)
                Xd.append(x_img)
            return np.array(Xd), self.Y[indexes,:]

    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

# Used by the Keras Model to report for us the AUC at each epoch and save the best model to disk.
class auc_Callback(keras.callbacks.Callback):
    def __init__(self,validation_generator,model_save_path,data=None,byTPRatFPR0=False):
        self.validation_generator=validation_generator
        self.max_v=-1
        self.path = model_save_path
        self.data = data
        self.byTPRatFPR0 = byTPRatFPR0
    def on_epoch_end(self, epoch, logs=None):
        if self.validation_generator is not None:
            Ytest=[]
            Ypred=[]
            for x,y in self.validation_generator:
                Ytest.append(y)
                Ypred.append(self.model.predict(x))
            Ytest=np.vstack(Ytest)#.astype(int)
            Ypred=np.vstack(Ypred)
        else:
            Ytest = self.data[1]
            Ypred = self.model.predict(self.data[0])

        fpr, tpr, threshold = metrics.roc_curve(Ytest[:,0], Ypred[:,0])
        roc_auc = metrics.auc(fpr, tpr)
        tpr_fpr0=tpr[np.where(fpr==0)[0][-1]]
        print('')
        print('AUC:',roc_auc,"TPR@FPR0:",tpr_fpr0)
        if self.byTPRatFPR0:
            v = tpr_fpr0
        else:
            v = roc_auc
        if v > self.max_v:
            self.max_v = v
            self.model.save(self.path)

# Evalaute model with a given validation generator (made with train_expert) OR prediction and their groundtruth labels
def eval_model(model, validation_generator=None, predictions=None, labels=None):
    from matplotlib import pyplot as plt
    Ytest=[]
    Ypred=[]
    
    if validation_generator is not None:
        for x,y in validation_generator:
            Ytest.append(y)
            Ypred.append(model.predict(x))
        Ytest=np.vstack(Ytest)#.astype(int)
        Ypred=np.vstack(Ypred)
    elif predictions is not None:
        Ytest=predictions
        Ypred=labels
    else:
        raise Exception('you must provide a generator or the predicitons+labels')

    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = metrics.roc_curve(Ytest[:,0], Ypred[:,0])
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('ROCR')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    print("TPR at FPR=0:",tpr[np.where(fpr==0)[0][-1]],"with threshold:",threshold[np.where(fpr==0)[0][-1]])

    thr=0.5
    with_signs = Ypred[Ytest[:,1]==1,0]
    acc_xr = np.sum(with_signs>thr)/len(with_signs) #errors on real signs
    
    withf_signs = Ypred[Ytest[:,1]==0,0]
    acc_wfs = np.sum(withf_signs>thr)/len(withf_signs) #succ detecting fake signs

    print("errors on real signs:",acc_xr)
    print("succ detecting fake signs:",acc_wfs)

    from sklearn.metrics import classification_report, confusion_matrix
    print('Confusion Matrix')
    print(confusion_matrix(np.argmax(Ytest, axis=1), np.argmax(Ypred, axis=1)))
    return Ytest, Ypred

# used to create experts in train_expert()
def create_cnn(width, height, depth,  regress=False, filters=(16, 32, 64)):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1

    # define the model input
    inputs = Input(shape=inputShape)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu",name="embed")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    if regress:
        x = Dense(1, activation="linear")(x)
    else:
        x = Dense(2, activation="softmax")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model

# used to train the combined (meta) model
def create_dnn(input_dim):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (input_dim,)

    # define the model input
    inputs = Input(shape=inputShape)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Dense(32)(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)

    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)
    
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)
    x = Dense(2, activation="softmax")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model

def get_embeddings(model,generator):
    if generator.dynamic_loading:
        Y=[]
        emeddings=[]
        for x,y in generator:
            Y.append(y)
            emeddings.append(model.predict(x))
        return np.vstack(emeddings), np.vstack(Y)
    else:
        return model.predict(generator.X), np.array(generator.Y)
    