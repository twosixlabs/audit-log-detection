from __future__ import division
import logging
import numpy as np
import sklearn
from sklearn.utils import class_weight
from sklearn.metrics import log_loss, roc_curve, auc, f1_score
import tempfile
import os
import time
import threading
import multiprocessing
import Queue
import basicutils
import gzip
import json
import copy
from keras.models import Sequential
from keras.utils.generic_utils import Progbar

class TrainingHistory(object):
    
    def __init__(self, save_dir=None, verbose=0, *args, **kwargs):
        self.verbose = verbose
        self.validation = None
        self.save_dir = save_dir
    
    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
    
    def on_train_begin(self, model):
        pass

    def on_batch_end(self, model, epoch, batch):
        pass
    
    def get_validation(self):
        return self.validation

    def on_epoch_end(self, model, epoch, loss):
        return False
    
class TrainingHistoryMultiClass(TrainingHistory):
    
    def __init__(self, *args, **kwargs):
        super(TrainingHistoryMultiClass, self).__init__(*args, **kwargs)
        self.A = None
        self.y = None
    
    def set_validation(self, A, y):
        self.A = A
        self.y = y

    def on_train_begin(self, model):

        self.validation = {}
        self.validation['epoch'] = [] 
        self.validation['time'] = []    
        self.validation['weighted_f1'] = []    

    def on_epoch_end(self, model, epoch):
        
        from basicutils import one_hot_to_label
        
        y_pred = model.predict(self.A)
        
        #get the probability
        y_label = one_hot_to_label(self.y)
        y_pred_label = one_hot_to_label(y_pred)

        weighted_f1 = f1_score(y_label, y_pred_label, average='weighted')  

        self.validation['time'].append(time.time())
        self.validation['epoch'].append(epoch)
        self.validation['weighted_f1'].append(weighted_f1)   
        
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
              
            #saving keras weights
            model.save_keras_weights(os.path.join(self.save_dir,'final_model_weights.dat'))
                
            #saving the model
            with gzip.GzipFile(os.path.join(self.save_dir, 'final_model_history.json.gz'), 'w') as f:
                    json.dump(self.get_validation(), f, indent=4, separators=(',', ': '), sort_keys=True)
     
        
        return False
    
class TrainingHistoryTestRocksDB(TrainingHistory):
    
    def __init__(self, read_doubles = False, save_model=False, save_roc=False, *args, **kwargs):
        super(TrainingHistoryTestRocksDB, self).__init__(*args, **kwargs)
        
        self.read_doubles_ = read_doubles
        self.A = None
        self.db = None
        self.names = None
        self.y = None
        self.save_model = save_model
        self.save_roc = save_roc
        
    def set_validation_rocksdb(self, db, names, y):
        self.db = db
        self.names = names
        self.y = y

    def set_validation(self, A, y):
        self.A = A
        self.y = y

    def on_train_begin(self, model):

        self.validation = {}
        self.validation['epoch'] = [] 
        self.validation['auc'] = []    
        self.validation['time'] = []    
        self.validation['log_loss'] = []    
        self.validation['roc'] = []    

    def on_epoch_end(self, model, epoch):
        
        log = logging.getLogger(__name__)

        if (self.db is None and self.A is None) or self.y is None:
            return False
        
        if self.db is not None:
            y_pred = model.predict_proba_rocksdb(self.db, self.names, read_doubles=self.read_doubles_)
        else:
            y_pred = model.predict_proba(self.A)
        
        #get roc curve
        fpr, tpr, _ = roc_curve(self.y, y_pred[:,1])        
        curr_auc = auc(fpr, tpr)
        roc_val = {}
        roc_val['fpr'] = fpr.tolist()
        roc_val['tpr'] = tpr.tolist()
        
        #unweighted log loss
        loss = log_loss(self.y, y_pred, eps=1e-8)
        
        self.validation['time'].append(time.time())
        self.validation['epoch'].append(epoch)
        self.validation['auc'].append(curr_auc)
        self.validation['log_loss'].append(loss)
        if self.save_roc:
            self.validation['roc'].append(roc_val)
        
        if self.verbose>0:
            log.info('Current epoch {0} validation results: AUC={1}, log_loss={2}.'.format(epoch, curr_auc, loss))
            
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
              
            #saving the model
            log.info('Saving the current model and roc to {0}.'.format(self.save_dir))
            if self.save_model:
                #with open(os.path.join(self.save_dir,'final_model.p'), 'w') as f:
                #    pickle.dump(model, f)

                #saving keras weights
                model.save_keras_weights(os.path.join(self.save_dir,'final_model_weights.dat'))
                
            #saving the model
            with gzip.GzipFile(os.path.join(self.save_dir, 'final_model_history.json.gz'), 'w') as f:
                    json.dump(self.get_validation(), f, indent=4, separators=(',', ': '), sort_keys=True)
                
        
        return False

class KerasPipelineClassifier(object):
   
    
    def __init__(self, optimizer='adam', loss='binary_crossentropy', batch_size=128, memory_batch_size=512, n_jobs = 1,
                 nb_epoch=100, shuffle=True,
                 validation_split=0, validation_data=None, callbacks=None,
                 verbose=0,class_weight='auto',background=True, *args, **kawargs):
        
        self.model_ = None
        self.optimizer = optimizer
        self.loss = loss
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.memory_batch_size = memory_batch_size
        self.nb_epoch = nb_epoch
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.callbacks = callbacks
        self.verbose = verbose     
        self.class_weight = class_weight
        self.class_actual_weight_ = None
        self.background = background
        
        self.curr_batch_size = self.batch_size
        
        self.n_features_ = None

        assert self.memory_batch_size>=self.batch_size    
        
    def set_func(self, callbacks=None):
        if callbacks is not None:
            self.callbacks = callbacks
        
    def get_func(self):
        return self.callbacks
    
    def batch_adjustment(self, A, y, w):

        #make sure outputing matrix array
        if isinstance(A, list):
            A = np.matrix(A)
            
        return A,y,w

    def matrix_load_into_queue(self,X,batch_idx,queue,load_func,y=None,w=None):
        A = load_func(X,batch_idx)
        
        #get y
        if y is not None:
            y = y[batch_idx,...]
        if w is not None:
            w = w[batch_idx,...]
            
        A,y,w = self.batch_adjustment(A,y,w)
        
        if self.verbose>2:
            log = logging.getLogger(__name__)
            log.info('Finished loading next batch. Load size is {}.'.format(len(batch_idx)))

        queue.put((A,y,w))
    
    
    def fit(self, X, y, w = None):

        assert X.shape[0]==y.shape[0], 'Sizes do not match.'

        return self.fit_general(X, y, w, basicutils.matrix_load)
    
    def fit_rocksdb(self, X, y, names, w=None, read_doubles=False):

        assert y.shape[0]==len(names), 'Sizes do not match.'

        return self.fit_general(X, y, w, lambda X,idx:  basicutils.matrix_load_rocksdb(X,idx,names,read_doubles))
    
    def fit_general(self, X, y, w, load_func):
        
        return self.update_general(X, y, w, load_func, True)
 
    def update(self, X, y, w=None):
        
        assert X.shape[0]==y.shape[0], 'Sizes do not match.'

        return self.update_general(X, y, w, basicutils.matrix_load, X.shape[0])
    
    def update_rocksdb(self, X, y, names, w=None, read_doubles=False):

        assert y.shape[0]==len(names), 'Sizes do not match.'

        return self.update_general(X, y, w, lambda X,idx:  basicutils.matrix_load_rocksdb(X,idx,names,read_doubles))

    def update_general(self, X, y, w, load_func, reset=False):
        
        log = logging.getLogger(__name__)
        
        data_size = y.shape[0]
          
        #figure out the weights
        if w is None and len(y.shape)==1:
            y_unique = np.unique(y)
            weights = class_weight.compute_class_weight(self.class_weight, y_unique, y)
            self.class_actual_weight_ = {}
            for i, y_val in enumerate(y_unique):
                self.class_actual_weight_[y_val] = weights[i]
                
            w = np.zeros(data_size)
            for i,v in enumerate(y):
                w[i] = self.class_actual_weight_[v]
        elif w is None:
            log.warn('Do not know how to make class weights for multidimensinal output. If neeed, specify weights directly. Assuming uniform weights.')
            w = np.ones(data_size)
        else:
            assert w.shape[0]==data_size, 'Weight size should match data size.'    

        if self.background:        
            queue = multiprocessing.Queue()
        else:
            queue = Queue.Queue()
        

        log.info('Starting to fit the NN model.')
        
        if self.callbacks is not None:
            for callback in self.callbacks:
                if callback is not None:
                    callback.on_train_begin(self)
                
        for epoch in xrange(self.nb_epoch):
            
            last_update = time.time()-1000
            start_time = time.time()

            #generate the progress bar
            if self.verbose>0:
                progbar = Progbar(data_size, width=80, verbose=self.verbose)

            #get random permutation
            p = np.random.permutation(range(data_size))            
                        
            #load the first batch
            batch_idx = p[0:self.memory_batch_size];
            self.matrix_load_into_queue(X, batch_idx, queue, load_func, y, w);
            X_batch,y_batch,w_batch = queue.get()
            
            if reset and epoch==0:
                
                n_features = self.get_dimensions(X_batch)
                log.info('Compiling the NN model with {} dimensions.'.format(n_features))
                self.generate_and_compile_model_(n_features)
            
            samples = 0
            for batch, i in enumerate(xrange(0, len(p), self.memory_batch_size)):
                
                #compute indicies for next batch
                next_start = i+len(batch_idx)
                next_end = min(len(p), next_start+self.memory_batch_size)
                if next_end>next_start:
                    #spin the thread up                
                    batch_idx_next = p[next_start:next_end];

                    #load data in background
                    thread = 0                   
                    if self.background:
                        thread = multiprocessing.Process(target=self.matrix_load_into_queue, args=(X,batch_idx_next,queue,load_func, y, w))
                        thread.start()
                else:         
                    batch_idx_next = None
                    thread = None
                
                #perform update
                loss = self.batch_update(X_batch, y_batch, w_batch)
                
                #increment the counter
                samples+= len(batch_idx)

                curr_update = time.time()
                if  self.verbose>0 and (curr_update-last_update>=0.5 or (samples)>=len(p)):
                    progbar.update(samples, [('Loss', loss)])
                    last_update = curr_update

                if self.callbacks is not None:
                    for callback in self.callbacks:
                        if callback is not None:
                            r = callback.on_batch_end(self, epoch+1, batch+1)

                #wait for the next load to happen                
                if thread is not None:
                    #if no background, load the data now
                    if not self.background:
                        self.matrix_load_into_queue(X, batch_idx, queue, load_func, y, w)
                    X_batch,y_batch,w_batch = queue.get()
                    
                    #if loading a background process, do a join
                    if self.background:
                        thread.join()
                    
                #now add the next batch
                batch_idx = batch_idx_next

            finish_time = time.time()-start_time
            if self.verbose>0:
                log.info('Finished epoch {}/{}. Time per epoch (s): {:0.2f}, Time per sample (s): {}.'.format(epoch+1, self.nb_epoch,finish_time,finish_time/len(p)))
            
            #process the end of epoch, and see if need to quit out
            quit_now = False
            if self.callbacks is not None:
                for callback in self.callbacks:
                    if callback is not None:
                        r = callback.on_epoch_end(self, epoch+1)
                        if r is not None and r is True:
                            quit_now = True
                        
            
            if quit_now:
                break    
        
        return self   
       
    def regenerate_model(self):

        self.generate_and_compile_model_(self.n_features_)
       
    def compile(self):
            
        self.model_.compile(optimizer=self.optimizer, loss=self.loss, metrics=None)
            
    def generate_and_compile_model_(self,n_features):
        
        self.n_features_ = n_features
        self.model_ = self.get_model(n_features)
        self.compile()
        
    def get_class_weight_(self, y):
        
        if self.class_weight is 'auto':
            #class_weights = None
            benign_weight = 0.5*len(y)/float(sum(y==0))
            malware_weight = 0.5*len(y)/float(sum(y==1))
            class_weight = { 0 : benign_weight, 1 : malware_weight}
        else:
            class_weight = None
 
        return class_weight
    
    def get_dimensions(self, X):
        
        dim = []
        if X is list:

            first = X[0]
            while first is list:
                dim.append(len(first))
                first = first[0]
            if isinstance(first, (np.ndarray, np.generic)):
                dim.extend(list(X.shape[1:]))
        else:
            dim = X.shape[1:]
            
        return dim
    
    def batch_update(self, X, y, sample_weights):

        assert y is not None, 'y cannot be null.'
        assert sample_weights is not None, 'sample_weights cannot be null.'

        #figure out if list
        is_list = type(X) is list
        
        if is_list:
            if len(X)==0:
                size = 0
            else:
                size = X[0].shape[0]
        else:
            size = X.shape[0]
            
        if size==0 or y.shape[0]==0:
            log = logging.getLogger(__name__)
            log.warn('Recieved batch of size {}. Skipping with 0 loss returned.'.format(y.shape[0]))
            return 0           
            
        assert y.shape[0] == sample_weights.shape[0], 'Label size {} does not match weights size {}.'.format(y.shape[0], sample_weights.shape[0])
        assert y.shape[0] == size, 'Label size {} does not match data size {}.'.format(y.shape[0], size)
            
        loss = np.nan
        for i in xrange(0, size, self.batch_size):
            
            next_start = i
            next_end = min(size, next_start+self.batch_size)
        
            if is_list:
                X_batch = []
                for M in X:
                    X_batch.append(M[next_start:next_end,...])
            else:
                X_batch = X[next_start:next_end,...]
                
            #get y
            y_batch = y[next_start:next_end,...]
            w_batch = sample_weights[next_start:next_end]
            
            #perform training
            self.curr_batch_size = next_end-next_start
            loss = self.model_.train_on_batch(X_batch, y_batch, sample_weight=w_batch)
            
            loss = np.ravel(loss)[0]
            
        
        return loss
        
    def get_model(self, num_features):
        
        assert False
        
        return None 
    
    def predict(self, X):
        return self.predict_general(X, X.shape[0], basicutils.matrix_load)

    def predict_rocksdb(self, X, names, read_doubles=False):

        return self.predict_general(X, len(names), lambda X,idx:  basicutils.matrix_load_rocksdb(X,idx,names,read_doubles))

    def predict_general(self, X, size, load_func):
        
        return self.predict_general_(self.model_,  X, size, load_func)
    
    def predict_general_(self, model, X, size, load_func):

        queue = Queue.Queue()

        #generate the progress bar
        if self.verbose>0:
            progbar = Progbar(size, width=80, verbose=self.verbose)

        batch_idx = range(min(size, self.memory_batch_size));
        self.matrix_load_into_queue(X, batch_idx, queue, load_func);
        X_batch,_,_ = queue.get()

        p = []
        samples = 0
        last_update = time.time()-1000
        for _, i in enumerate(xrange(0, size, self.memory_batch_size)):
            
            next_start = i+len(batch_idx)
            next_end = min(size, next_start+self.memory_batch_size)
            if next_end>next_start:
                #spin the thread up                
                batch_idx_next = range(next_start,next_end);
                
                thread = threading.Thread(target=self.matrix_load_into_queue, args=(X,batch_idx_next,queue,load_func))
                thread.start()
            else:         
                batch_idx_next = None
                thread = None

            #predict the value
            if X_batch.shape[0]>0:
                p_curr = model.predict(X_batch, batch_size=self.batch_size, verbose=0)
                p.append(p_curr)
            
            #increment the counter
            samples+= len(batch_idx)
            
            curr_update = time.time()
            if  self.verbose>0 and (curr_update-last_update>=0.5 or (samples)>=size):
                progbar.update(samples, [])
                last_update = curr_update
                    
            #wait for the next load to happen                
            if thread is not None:
                thread.join()
                X_batch,_,_ = queue.get()
                
            #now add the next batch
            batch_idx = batch_idx_next
        
        p = np.vstack(p)
        
        return p
    
    def change_loss(self, loss='mse'):
                
        gen_model = self.get_model(self.n_features_)
        layers = gen_model.layers
                
        model = Sequential(layers=layers)
        
        log = logging.getLogger(__name__)
        log.info('Compiling model with new loss function.')
        
        model.compile(self.optimizer, loss)
        
        #now set the weights
        for i in xrange(len(model.layers)):
            model.layers[i].set_weights(self.model_.layers[i].get_weights())
                
        self.model_ = model
        
        return self
    
    def copy_weights(self, model, layers):
        
        for i in layers:
            self.model_.layers[i].set_weights(model.model_.layers[i].get_weights())
        
    
    def get_submodel(self, num_layers=None, loss='mse'):
                
        gen_model = self.get_model(self.n_features_)

        if num_layers is None:
            num_layers = len(gen_model.layers) 
                
        layers = gen_model.layers[:num_layers]
        
        model = Sequential(layers=layers)
        model.compile(self.optimizer, loss)
        
        #now set the weights
        for i in xrange(len(model.layers)):
            model.layers[i].set_weights(self.model_.layers[i].get_weights())
            
        #generate a copy and attach the weights
        model_copy = copy.deepcopy(self)        
        model_copy.model_ = model
        model_copy.loss = loss
        
        return model_copy
    
    def predict_proba(self, X):
        
        return self.predict_proba_general(X, X.shape[0], basicutils.matrix_load)

    def predict_proba_rocksdb(self, X, names, read_doubles=False):

        return self.predict_proba_general(X, len(names), lambda X,idx:  basicutils.matrix_load_rocksdb(X,idx,names,read_doubles))

    def predict_proba_general(self, X, size, load_func):

        preds = self.predict_general(X, size, load_func)
        
        if len(preds.shape)==1:
            preds = np.array([preds]).T
        
        if preds.shape[1]==1:
            p_neg = 1-preds
            preds = np.hstack((p_neg, preds))
            
        #normalize
        preds = sklearn.preprocessing.normalize(preds, norm='l1', axis=1)
        
        assert preds.min() >= 0 and preds.max() <= 1, 'Network returning invalid probability values.'
            
        return preds 
    
    def save_keras_weights(self, f):
        self.model_.save_weights(f, overwrite=True)        
    
    def load_keras_weights(self, f):
        self.model_.load_weights(f)        

    def get_model_weights_(self):
        
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.close()
        data = None
        
        try:        
            self.save_keras_weights(temp.name)
            with open(temp.name, 'rb') as f:
                data = f.read()            
            
        finally:
            os.remove(temp.name)
            
        return data
    
    def load_model_weights_(self, data):
        
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.close()
        
        try:        
            with open(temp.name, 'wb') as f:
                f.write(data)
                f.close()
                            
            self.load_keras_weights(temp.name)
            
        finally:
            os.remove(temp.name)
            
        return data
    
    def print_layers(self):
        print('Generated model:')
        for i in xrange(len(self.model_.layers)):
            print('\tLayer {}: {}'.format(i, str(self.model_.layers[i])))
         
    def __getstate__(self):
        d = self.__dict__.copy()
         
        #log.info("Calls pickle")
         
        #store the weights
        del d['callbacks']        
        if self.model_ is not None:
            d['weights'] = self.get_model_weights_()
            del d['model_']
        else:
            d['weights'] = None
          
        return d
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Dictionary of parameter names mapped to their values.
        """
        model_val = None
        if self.model_ is not None:
            model_val = json.loads(self.model_.to_json())
        
        return {'model': model_val, 'optimizer': self.optimizer, 'loss': self.loss, 'epoch': self.nb_epoch}
      
    def __setstate__(self, d):
          
        data = d['weights']
        del d['weights']
          
        self.__dict__.update(d) 
        
        #now load the weights
        if data is not None:
            self.generate_and_compile_model_(self.n_features_)
            self.load_model_weights_(data)
