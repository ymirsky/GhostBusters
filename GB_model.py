# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import time

import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import AUC
import sklearn.metrics as metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

from utils import *


class GhostBusters:
    def __init__(self, model_path=None, save_path="models", device_ID=None):
        # model_path: provide path to parent directory containing the models (context, surface, light, optical, combined)
        # Device ID: the GPU index to use. Set to "" for CPU and leave as None for default
        if device_ID is not None:
            os.environ["CUDA_VISIBLE_DEVICES"]=device_ID
            
        self.save_path = save_path
        self.real_dir_path = ''
        self.fake_dir_path = ''
        self.nosign_dir_path = ''
        self.expert_names=np.array(["context","surface","light","optical"])
        self.train_gens = {"context":None, "surface":None, "light":None, "optical":None}
        self.validation_gens = {"context":None, "surface":None, "light":None, "optical":None}
        self.experts = {"context":None, "surface":None, "light":None, "optical":None}
        self.embeddings = {"context":None, "surface":None, "light":None, "optical":None}
        self.combiner = None
        self.experts_trained = False #experts trained?
        self.is_trained = False #all models trained?
        
        # Load saved models?
        if model_path is not None:
            for file in os.listdir(model_path):
                for expert in self.expert_names:
                    if expert in file:
                        model = keras.models.load_model(os.path.join(model_path,file))
                        self.embeddings[expert] = Model(inputs=model.input,outputs=model.layers[19].output)
                if 'combiner' in file:
                    self.combiner = keras.models.load_model(os.path.join(model_path,file))
            #check if all loaded
            for expert in self.expert_names:
                if self.embeddings[expert] is None:
                    raise Exception('Could not load '+expert+' model.')
            if self.combiner == None:
                raise Exception('Could not load combiner model.')
            self.experts_trained = True #experts trained?
            self.is_trained = True #all models trained?
            print('Loaded all models successfully.')
                        

    def train(self,real_dir_path,fake_dir_path,nosign_dir_path, epochs=[2,2]):
        # {real, fake}_dir_paths are directories containing images detected by the roadsign detector in npy format, organized by expert. 
        # for training the context model, fake_paths should be to exampels or where there are no sign (training) and context_val_path should be the path to the phantom examples (validation)
        # Return: trained models and the train/validation data generators
        self._train_experts(real_dir_path,fake_dir_path,nosign_dir_path,epochs=epochs[0])
        self._train_comittee(epochs=epochs[1])
        
    def _train_experts(self,real_dir_path,fake_dir_path,nosign_dir_path,epochs=25):
        # {real, fake}_dir_paths are directories containing images detected by the roadsign detector in npy format, organized by expert. 
        #         E.g., if real_dir_path="data/real", then there should be subdirectories "data/real/context", "data/real/surface" ...etc that have the corrisponding npy images inside
        # for training the context model, fake_paths should be to exampels or where there are no sign (training) and context_val_path should be the path to the phantom examples (validation)
        # Return: trained models and the train/validation data generators
        self.real_dir_path = real_dir_path
        self.fake_dir_path = fake_dir_path
        self.nosign_dir_path = nosign_dir_path
        
        for expert in self.expert_names:
            real_path_e = os.path.join(self.real_dir_path,expert)
            fake_path_e = os.path.join(self.fake_dir_path,expert)
            nosign_path_e = os.path.join(self.nosign_dir_path,expert)
            
            # prep data paths for each expert
            Xtrain,Ytrain,Xtest,Ytest = split_data(real_path_e, fake_path_e)
            if expert == 'context':
                Xtrain,Ytrain,_,_ = split_data(real_path_e, nosign_path_e)
            if expert == 'light':
                dim = [88,88,1]
            elif expert == 'surface':
                dim = [88,88,3]
            else:
                dim = [128,128,3]
                        
            # Make the data loader (for dynamically loading from disk)
            self.train_gens[expert] = DataLoader(Xtrain,Ytrain,batch_size=16,shuffle=True,dynamic_loading=False)
            self.validation_gens[expert] = DataLoader(Xtest,Ytest,batch_size=16,shuffle=True,dynamic_loading=False)

            ##### Train Expert ######
            print("[INFO] Training",expert,"Model...")
            opt = Adam(lr=1e-3, decay=1e-3 / 200)
            model = create_cnn(dim[0], dim[1], dim[2])
            model.compile(loss="categorical_crossentropy", optimizer=opt)
            model.summary()

            model_path = os.path.join(self.save_path,expert)
            mcp_save = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
            model.fit(self.train_gens[expert],
                        validation_data=self.validation_gens[expert],
                        epochs=epochs, callbacks=[mcp_save])
            try:
                model = keras.models.load_model(model_path) #load best model (last one saved to disk)
            except:
                print("ERROR: could not load model")
                
            self.experts[expert] = model
            self.embeddings[expert] = Model(inputs=model.input,outputs=model.layers[19].output)
            self.experts_trained = True
        print("[INFO] Done training experts.")

    # Train Combined model
    def _train_comittee(self,epochs=10):
        if not self.experts_trained:
            raise Exception("You must first train the experts")
            
        ### Extract the embeddings ###
        #Load with sign data for context model:
        real_path_e = os.path.join(self.real_dir_path,'context')
        fake_path_e = os.path.join(self.fake_dir_path,'context')
        Xtrain,Ytrain,Xtest,Ytest = split_data(real_path_e, fake_path_e)
        train_gen_c = DataLoader(Xtrain,Ytrain,batch_size=16,shuffle=True,dynamic_loading=False)
        
        print('[INFO] Extracting Embeddings')
        Xtrain_ce,Ytrain_ce = get_embeddings(self.embeddings['context'],train_gen_c)
        Xtrain_se,Ytrain_se = get_embeddings(self.embeddings['surface'],self.train_gens['surface'])
        Xtrain_le,Ytrain_le = get_embeddings(self.embeddings['light'],self.train_gens['light'])
        Xtrain_oe,Ytrain_oe = get_embeddings(self.embeddings['optical'],self.train_gens['optical'])
        print(Xtrain_ce.shape,Xtrain_se.shape,Xtrain_le.shape,Xtrain_oe.shape)
        Xtrain_csloe = np.hstack((Xtrain_ce,Xtrain_se,Xtrain_le,Xtrain_oe))
        assert(np.array_equal(Ytrain_ce,Ytrain_oe))
        Ytrain_csloe = Ytrain_ce
        
        Xtest_ce,Ytest_ce = get_embeddings(self.embeddings['context'],self.validation_gens['context'])
        Xtest_se,Ytest_se = get_embeddings(self.embeddings['surface'],self.validation_gens['surface'])
        Xtest_le,Ytest_le = get_embeddings(self.embeddings['light'],self.validation_gens['light'])
        Xtest_oe,Ytest_oe = get_embeddings(self.embeddings['optical'],self.validation_gens['optical'])
        Xtest_csloe = np.hstack((Xtest_ce,Xtest_se,Xtest_le,Xtest_oe))
        assert(np.array_equal(Ytest_ce,Ytest_oe))
        Ytest_csloe = Ytest_ce

        ### Train Model ###
        # Train
        print("[INFO] Training Combined Model...")
        opt = Adam(lr=1e-3, decay=1e-3 / 200)
        self.combiner = create_dnn(Xtrain_csloe.shape[1])
        self.combiner.compile(loss="categorical_crossentropy", optimizer=opt)
        self.combiner.summary()

        model_path = os.path.join(self.save_path,'combiner')
        mcp_save = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
        self.combiner.fit(x=Xtrain_csloe,y=Ytrain_csloe,validation_data=(Xtest_csloe,Ytest_csloe),shuffle=True,epochs=epochs,callbacks=[mcp_save])
        self.combiner = keras.models.load_model(model_path)# load best model which was saved to disk
        self.is_trained = True
        print('[INFO] Done training combiner.')
        
    def predict(self,path=None):
        # paths: list of paths to the npy images of road signs (real or fake)
        #            {real, fake}_paths are directories containing images detected by the roadsign detector in npy format, organized by expert. 
        #            E.g., if real_dir_path="data/real", then there should be subdirectories "data/real/context", "data/real/surface" ...etc that have the corrisponding npy images inside
        #     Leave as None if you want to use the same validation set from training
        
        if not self.is_trained:
            raise Exception("You must first train the experts and the combiner models")
        
        ### Prep Data Loaders (generators) ###
        generators = {"context":None, "surface":None, "light":None, "optical":None}
        if path is None:
            generators = self.validation_gen
        else:
            for expert in self.expert_names:
                path_e = os.path.join(path,expert)
                X = [os.path.join(path_e,file) for file in os.listdir(path_e)]

                # Make the data loader (for dynamically loading from disk)
                generators[expert] = DataLoader(X,[np.nan]*len(X),batch_size=16,shuffle=True,dynamic_loading=False)
            
        ### Execute Experts - get their embeddings ###
        s = time.time()
        X_ce,Y_ce = get_embeddings(self.embeddings['context'],generators['context'])
        X_se,Y_se = get_embeddings(self.embeddings['surface'],generators['surface'])
        X_le,Y_le = get_embeddings(self.embeddings['light'],generators['light'])
        X_oe,Y_oe = get_embeddings(self.embeddings['optical'],generators['optical'])
        X_csloe = np.hstack((X_ce,X_se,X_le,X_oe))
        
        ### Execute Combiner ###
        pred = self.combiner.predict(X_csloe)
        f = time.time()
        
        print(len(Y_oe),"samples took",np.round(f-s,3),"seconds.")
        return pred
        

