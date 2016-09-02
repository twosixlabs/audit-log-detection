from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Lambda
from keras.layers.advanced_activations import PReLU
#from keras.layers.normalization import BatchNormalization
from utils.KerasBatchNormalization import BatchNormalization
import logging
from utils.KerasWrapper import KerasPipelineClassifier
from keras import backend as K
from keras import objectives
from keras.layers import Embedding
from keras.layers import Convolution1D
from keras.layers import LSTM, SimpleRNN, GRU
import logging
import numpy as np

from utils.KerasWrapper import KerasPipelineClassifier

log = logging.getLogger('invincea')

def seq_batch_adjustment(rocks_counter,A,y,w):
        
        vnew = []
        for vals in A:
            
            top_vals = [rocks_counter.index(v) for v in vals if rocks_counter.index(v)>=0]
            
            #remove repeats
            new_vals = []
            for i in xrange(len(top_vals)):
                if i==0 or new_vals[-1] != top_vals[i]:
                    new_vals.append(top_vals[i])
            top_vals = new_vals
                                    
            #now pad with zeros
            while len(top_vals)<rocks_counter.max_length:
                top_vals.append(-1)
                
            #get results
            top_vals = top_vals[:rocks_counter.max_length]
                
            vnew.append(top_vals)
            
        A = np.matrix(vnew, dtype=np.float32)
        
        #make it so padding is zero
        A = A+1.0

        return A,y,w    


class AuditNN(KerasPipelineClassifier):

    def __init__(self, *args, **kwargs):
        super(AuditNN, self).__init__(*args, **kwargs)
        
    def get_model(self, num_features):
        
        model = Sequential()

        model.add(Dropout(0.2,input_shape=(num_features,)))
        model.add(Dense(1024, init='glorot_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Dropout(0.2))
        model.add(Dense(1024, init='glorot_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Dropout(0.2))
        model.add(Dense(1024, init='glorot_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        return model
    
class AuditRNN(KerasPipelineClassifier):

    def __init__(self, rocks_counter, vocab_size=128, *args, **kwargs):
        super(AuditRNN, self).__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.rocks_counter = rocks_counter
        
    def batch_adjustment(self,A,y,w):
        
        return seq_batch_adjustment(self.rocks_counter,A,y,w) 
        
    def get_model(self, num_features):
        
        embedding_dims = 128
        lstm_output_size = 128;
        
        drop = 0.2
           
        model = Sequential()
        
        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        model.add(Embedding(self.vocab_size,
                            embedding_dims,
                            input_length=num_features[0],
                            dropout=0.2))
        
        model.add(LSTM(output_dim=lstm_output_size, go_backwards=True))
        
        model.add(Dropout(drop))
        model.add(Dense(1024, init='glorot_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dropout(drop))
        model.add(Dense(1024, init='glorot_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        return model

class AuditConv(KerasPipelineClassifier):

    def __init__(self, rocks_counter, vocab_size=128, *args, **kwargs):
        super(AuditConv, self).__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.rocks_counter = rocks_counter
        
    def batch_adjustment(self,A,y,w):
        
        return seq_batch_adjustment(self.rocks_counter,A,y,w) 
        
    def get_model(self, num_features):
        
        embedding_dims = 128
        nb_filter = 250
        filter_length = 8
        
        drop = 0.2
           
        model = Sequential()
        
        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        model.add(Embedding(self.vocab_size,
                            embedding_dims,
                            input_length=num_features[0],
                            dropout=0.2))
        
        # we add a Convolution1D, which will learn nb_filter
        # word group filters of size filter_length:
        model.add(Convolution1D(nb_filter=nb_filter,
                                filter_length=filter_length,
                                border_mode='valid',
                                activation='relu',
                                subsample_length=1))
        
        # we use max over time pooling by defining a python function to use
        # in a Lambda layer
        def max_1d(X):
            return K.max(X, axis=1)
        
        model.add(Lambda(max_1d, output_shape=(nb_filter,)))
        
        model.add(Dropout(drop))
        model.add(Dense(1024, init='glorot_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dropout(drop))
        model.add(Dense(1024, init='glorot_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        return model
    