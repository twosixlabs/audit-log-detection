from __future__ import division
import logging
import numpy as np
import scipy
import struct
import collections
import string

log = logging.getLogger(__name__)

def flatten_json(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def strings(s, min_length=4):
    result = ""
    for c in s:
        if c in string.printable:
            result += c
            continue
        if len(result) >= min_length:
            yield result
        result = ""

def strings_file(filename, min_length=4):
    with open(filename, "rb") as f:
        result = ""
        for c in f.read():
            if c in string.printable:
                result += c
                continue
            if len(result) >= min_length:
                yield result
            result = ""

def matrix_load(X,batch_idx):
    if scipy.sparse.issparse(X):
        X_batch = X[batch_idx,:].todense()
    else:
        X_batch = X[batch_idx,:]
        
    return X_batch


def one_hot_to_label(y):
    
    return np.ravel(np.argmax(y, axis=1))


def matrix_load_rocksdb(rocks_db, batch_idx, names_all, read_doubles = False):

    step_size = 100

    A = []
    
    for i in xrange(0, len(batch_idx), step_size):
        
        #get the files
        local_names = [names_all[x] for x in batch_idx[i:(i+step_size)]]
        
        entry_vals = rocks_db.multi_get(local_names)
        
        for name in local_names:
            val = entry_vals[name]
            assert val is not None, 'Could not find feature values for key {}'.format(name)
            
            if read_doubles:
                vals = np.array(struct.unpack(">{0}d".format(int(len(val)/8)), val))
            else:
                vals = np.array(struct.unpack(">{0}f".format(int(len(val)/4)), val))
            
            assert np.isfinite(vals).all(), 'Failed to get finite data for {0}.\n{1}'.format(str(name), [(i,x) for i,x in enumerate(vals) if not np.isfinite(x)])
            
            #remove from memory
            entry_vals[name] = None
            
            #append to storage
            A.append(vals)
            
        
    return A

class RocksCounter(object):
    
    def __init__(self, num_vocab, read_doubles=False, max_length=None, verbose=0):
        self.verbose = verbose
        self.read_doubles = read_doubles
        self.num_vocab = num_vocab
        self.counts = None
        self.max_length = max_length
        self.features = None
    
    def store_counts(self, rocksdb, names, features=None, starts_with_filter=None):

        #get the unique counts    
        vec_iter = RocksDBVectorIterator(rocksdb, names, self.read_doubles, self.verbose)
        counter = collections.Counter()
        for v in vec_iter:
            v = set(v)
            
            #remove vectors not used
            if starts_with_filter is not None and features is not None:
                v_new = []
                for i in v:
                    if not features[int(i)].startswith(starts_with_filter):
                        v_new.append(i)
                v = v_new
            
            counter.update(v)
            
        vals = counter.most_common(self.num_vocab)
        self.features = vals
        
        if features is not None:
            self.features = []
            for v,c in vals:
                self.features.append((features[int(v)],c))
        
        c = 0
        self.counts = {}
        for i,_ in vals:
            self.counts[i] = c
            c+=1
             
    def index(self, v):
        if v in self.counts:
            return self.counts[v]        
        
        return -1
    
class RocksDBVectorIterator(object):
    def __init__(self, A_rocksdb, names, read_doubles, verbose=0):
        self.db = A_rocksdb
        self.names = names
        self.read_doubles = read_doubles
        self.max_length = 0
        
    def load(self, name):
        A = matrix_load_rocksdb(self.db, [0], [name], self.read_doubles)
        return np.ravel(A)
    
    def __iter__(self):
        for _, name in enumerate(self.names):
            A = matrix_load_rocksdb(self.db, [0], [name], self.read_doubles)
            vect = np.ravel(A)
            
            yield vect

class RocksDBValueIterator(object):
    def __init__(self, A_rocksdb, names, read_doubles, verbose=0):
        self.db = A_rocksdb
        self.names = names
        self.read_doubles = read_doubles
        self.max_length = 0
        
    def load(self, name):
        A = matrix_load_rocksdb(self.db, [0], [name], self.read_doubles)
        return np.ravel(A)
    
    def __iter__(self):
        for i, name in enumerate(self.names):
            A = matrix_load_rocksdb(self.db, [0], [name], self.read_doubles)
            vect = np.ravel(A)
            
            for v in vect:
                if v>=0:
                    yield v
            
            if (i+1)%10000==0:
                log.info('Iterated through {0} of of {1} samples in the database.'.format(i+1, len(self.names)))
    