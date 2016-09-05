from __future__ import division
import matplotlib
from __builtin__ import dict
matplotlib.use('Agg')
import os
import logging
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.pipeline import Pipeline
import sys
import numpy as np
import scipy
import datetime
import time
from utils import validation
from sklearn.preprocessing import Binarizer, StandardScaler, Normalizer
import sqlite3
import rocksdb
import struct
from models.nn_models import AuditRNN, AuditConv
from utils.OccurrenceFilter import HashTrick
from utils.OccurrenceFilter import CountThreshold
import traceback
from utils import basicutils
import collections
import baker
import zlib

logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
log = logging.getLogger(__name__)

def load_time_data(path, data):
    
    log.info('Reading in the time feature matrix file.')

    db = rocksdb.DB(os.path.join(path,'auditlog_linear_time_rocksdb'), rocksdb.Options(create_if_missing=False), read_only=True)
    data['db_time'] = db
    
    log.info('Finished reading in the feature file.')
    
    if 'features' in data:
        for i,feat in enumerate(data['features']):
            assert feat and feat!='{}', 'Feature %d has empty name.' % i
    
    return data

def load_data(path, size, data={}, read_features = True):
    
    features = []
    if read_features:
        log.info('Reading in the features names file.')
        
        db = rocksdb.DB(os.path.join(path,'auditlog_features_rocksdb'), rocksdb.Options(create_if_missing=False), read_only=True)
        it = db.itervalues()
        it.seek_to_first()
        #s = struct.Struct(">q")
        for name in it:
            #val = s.unpack(idx)[0]
            features.append(name)
    data['features'] = features
    
    log.info('Reading in the meta data file.')

    conn = sqlite3.connect(os.path.join(path,'auditlog_meta.db'));
    
    c = conn.cursor()
    names = []
    vt_score = []
    source = []
    time_seen = []
    for row in c.execute('SELECT name, vt_score, source, best_time_first_seen FROM samples'):
        
        name = row[0]
        names.append(name)
        vt_score.append(row[1])
        source.append(row[2])
        time_seen.append(row[3])
        
    #now filter by size
    if size>0 and len(names)>=size:
        #create a random permutation
        p = np.random.permutation(range(len(names)))
        
        #get the random subset
        names = [names[p[i]] for i in xrange(size)]
        vt_score = [vt_score[p[i]] for i in xrange(size)]
        source = [source[p[i]] for i in xrange(size)]
        time_seen = [time_seen[p[i]] for i in xrange(size)]

    log.info('Loaded {} enteries from the Sqlite3 DB.'.format(len(names)))
        
    data['names'] = names
    data['score'] = np.array(vt_score)
    data['source'] = source
    data['time_seen'] = np.array(time_seen)
        
    db = rocksdb.DB(os.path.join(path,'auditlog_matrix_rocksdb'), rocksdb.Options(create_if_missing=False), read_only=True)    
    data['db_matrix'] = db
        
    if 'features' in data:
        for i,feat in enumerate(data['features']):
            assert feat and feat!='{}', 'Feature %d has empty name.' % i
    
    return data

def form_feature_matrix(db, names, max_size=None):
    
    log.info('Extracting feature matrix from db file.')
    
    A_list = basicutils.matrix_load_rocksdb(db, range(len(names)), names, read_doubles = True)
    
    col0 = []
    col1 = []
    col2 = []
    count = 0
    for vals in A_list:
        
        for i in xrange(int(len(vals)/2)):        
            col0.append(count)
            col1.append(vals[2*i+0])
            col2.append(vals[2*i+1])
        
        count+=1
  
    if (max_size==None):
        A = scipy.sparse.csc_matrix((col2, (col0, col1)))
    else:
        A = scipy.sparse.csr_matrix((col2, (col0, col1)), shape=[count,max_size], dtype=np.float32)
        
    return A

def get_labels(data, cutoff, valid_idx=None):
    
    if valid_idx is None:
        valid_idx = range(len(data['source']))
    
    y = []
    trusted_source = ['ENTERPRISE_SPLUNK']
    for i, idx in enumerate(valid_idx):
        score = data['score'][idx]
        source = data['source'][idx]
        
        if score<=1.0/55.0 or source in trusted_source:
        #if score==0:
            y.append(0)
        elif score>=cutoff:
            y.append(1)
        elif score>0 and score<cutoff:
            y.append(-2)
        elif score==-1:
            y.append(-3)
        else:
            y.append(np.NaN)
        
    return np.array(y)

def valid_by_types(data, valid_idx):
    
    cuckoo_idx = []
    splunk_idx = []
    
    for i in valid_idx:
        if data['source'][i] in ['CUCKOO']:
            cuckoo_idx.append(i)
        elif data['source'][i] in ['ENTERPRISE_SPLUNK']:
            splunk_idx.append(i)
        
    return (cuckoo_idx, splunk_idx)    

def cv_valid(data, cutoff, folds, make_syn):
    
    log.info('Creating CV splits.')
    
    y = get_labels(data,cutoff)
    valid_idx = []
    for i,y_val in enumerate(y):
        if y_val==0 or y_val==1:
            valid_idx.append(i)
            
    log.info('Data label distribution: total={0}, benign={1}, malware={2}, ambigious={3}, client_unlabeled={4}, no_vt={5}, unknown={6}.'.format(len(y), np.sum(y==0), np.sum(y==1), np.sum(y==-2), np.sum(y==-1), np.sum(y==-3), np.sum(np.isnan(y))))

    first_seen = data['time_seen']
    cuckoo_idx, splunk_idx = valid_by_types(data, valid_idx)
    
    log.info('Time split stats: min(days)={}, max(days)={}, std(days)={}.'.format(np.min(first_seen[cuckoo_idx])/86400.0, np.max(first_seen[cuckoo_idx])/86400.0, np.std(first_seen[cuckoo_idx])/86400.0))
    
    #first seen time
    time_cut = np.median(first_seen[valid_idx])
    train = []
    test = []
    for i in cuckoo_idx:
        if first_seen[i]<time_cut:
            train.append(i)
        else:
            test.append(i)
            
    cv_time = [[np.array(train), np.array(test)]]
    
    cv_cuckoo = StratifiedKFold(y[cuckoo_idx], n_folds=folds, shuffle=True)
    if len(splunk_idx)>=folds:
        cv_splunk = KFold(len(splunk_idx), n_folds=folds+1, shuffle=True)
    else:
        cv_splunk = []
        for i in xrange(folds+1):
            cv_splunk.append([[],[]])
    
    #touples
    syn_touples = []
    cv_sandbox = []
    cv_enterprise = []
    count = 1
    for cuckoo,splunk in zip(cv_cuckoo, cv_splunk):
        train = []
        test_sandbox = []
        test_enterprise = []
        
        train.extend([cuckoo_idx[i] for i in cuckoo[0]])
        train.extend([splunk_idx[i] for i in splunk[0]])
        
        test_sandbox.extend([cuckoo_idx[i] for i in cuckoo[1]])
        test_enterprise.extend([splunk_idx[i] for i in splunk[1]])
        
        #add the malware from cuckoo box
        test_enterprise.extend([cuckoo_idx[i] for i in cuckoo[1] if y[cuckoo_idx[i]]==1])
                    
        train = np.array(train)
        curr_sandbox = [train,np.array(test_sandbox)]
        curr_enterprise = [train,np.array(test_enterprise)]
        cv_sandbox.append(curr_sandbox)
        cv_enterprise.append(curr_enterprise)
        
        cuckoo_idx_c, splunk_idx_c = valid_by_types(data, curr_sandbox[0]) 
        cuckoo_idx_d, splunk_idx_d = valid_by_types(data, curr_sandbox[1]) 
        
        log.info('Created sandbox split %d: training size=%d (benign=%d,malware=%d,cuckoo=%d,splunk=%d), testing size=%d (benign=%d,malware=%d,cuckoo=%d,splunk=%d).' % (count, len(curr_sandbox[0]), np.sum(y[curr_sandbox[0]]==0), np.sum(y[curr_sandbox[0]]==1), len(cuckoo_idx_c), len(splunk_idx_c), len(curr_sandbox[1]), np.sum(y[curr_sandbox[1]]==0), np.sum(y[curr_sandbox[1]]==1), len(cuckoo_idx_d), len(splunk_idx_d)))
        if len(curr_enterprise[1])>0: 
            log.info('Created enterprise split %d: training size=%d (benign=%d,malware=%d), testing size=%d (benign=%d,malware=%d).' % (count, len(curr_enterprise[0]), np.sum(y[curr_enterprise[0]]==0), np.sum(y[curr_enterprise[0]]==1), len(curr_enterprise[1]), np.sum(y[curr_enterprise[1]]==0), np.sum(y[curr_enterprise[1]]==1)))
        
        count+=1

    if make_syn:
        splunk_count = 0
        for train,test in cv_splunk:
            if splunk_count==folds:
                syn_touples.extend(test)
            splunk_count+=1
        syn_touples = np.array(syn_touples)
    
    return cv_sandbox, cv_enterprise, syn_touples, cv_time

def get_full_time_data(data, cutoff, syn_touples=None):
    
    log.info('Generating the full dataset.')
    
    A = data['db_time']
    names = data['names']    
    y = get_labels(data,cutoff)
    
    return A,y,names
        
def get_ngram_data(data, cutoff, syn_touples=None):
    
    log.info('Generating the full dataset.')
    
    names = data['names']    
    A = form_feature_matrix(data['db_matrix'], names)
    y = get_labels(data,cutoff)
    
    A_graft = None
    if syn_touples is not None or len(syn_touples)>0:
        A_graft = A[syn_touples,:]
        
    return A,y,names,A_graft

@baker.command
def print_log_example(path):
    
    #features = []
    log.info('Reading in one json sample.')
    
    db = rocksdb.DB(os.path.join(path,'auditlog_json_rocksdb'), rocksdb.Options(create_if_missing=False), read_only=True)
    it = db.iteritems()
    it.seek_to_first()
    for name,v in it:
        
        v = zlib.decompress(v, 16+zlib.MAX_WBITS)
        
        print name, v
        
        break   

@baker.command
def run(input_dir, results_dir, model = "full", epoch=100, size=-1, cv_folds = 2, cutoff = 0.3):
    
    cv_folds = int(cv_folds)
    cutoff = float(cutoff) 
    epoch = int(epoch)
    size = int(size)
    
    data = load_data(input_dir, size, read_features=True);    
    data = load_time_data(input_dir, data);
    
    log.info('Finished reading data.')
    
    cv_cuckoo, cv_enterprise, syn_touples, cv_time = cv_valid(data, cutoff, cv_folds, False)
    
    feature_names = data['features']
    
    if model in ["ngram", "ngram_lr", "full"]:
        A_gram,y_gram,names_gram,_ = get_ngram_data(data, cutoff, syn_touples)
        A_graft_gram = None
    db_time,y,names = get_full_time_data(data, cutoff, syn_touples)
    
    log.info('Found {0} features.'.format(len(feature_names)))
        
    log.info('Finished filtering and splitting data data.')
    voc_size = 1000

    #counter
    feature_counter = basicutils.RocksCounter(voc_size, read_doubles=True, max_length=5000, verbose=1)

    ngram_transform_pipe = Pipeline([('empty', CountThreshold(count=2)),('transform', Binarizer(threshold=0.0)),('hash', HashTrick(n_features=10000))])
    ngram_transform_pipe_lr = Pipeline([('empty', CountThreshold(count=2)),('transform', Binarizer(threshold=0.0))])

    modelRF = RandomForestClassifier(n_estimators=128, n_jobs=-1, verbose=True,bootstrap=True,oob_score=False,min_samples_leaf=3, class_weight='balanced');
    modelLR = LogisticRegressionCV(penalty='l1',solver='liblinear', Cs=np.logspace(-3,2, 20), verbose=1, n_jobs=-1, class_weight='balanced', cv=10, scoring='log_loss', max_iter=200)
        
    modelConv =     AuditConv(feature_counter, class_weight='balanced', batch_size=64, optimizer='adam', vocab_size=voc_size+1, verbose=1, nb_epoch=epoch, memory_batch_size=64, callbacks=None)

    #the rnn model is experimental
    modelRNN =       AuditRNN(feature_counter, class_weight='balanced', batch_size=64, optimizer='adam', vocab_size=voc_size+1, verbose=1, nb_epoch=epoch, memory_batch_size=64, callbacks=None)

    final_model_ngram_lr = modelLR
    final_model_ngram = modelRF
    final_model_rnn = modelRNN
    final_model_conv = modelConv
   
    train_func_ngram = lambda model, A, y, idx, validation_idx: train_ngram(model, A, y, idx, None, names, ngram_transform_pipe);
    predict_func_ngram = lambda model, A, idx: predict_ngram(model, A, idx, ngram_transform_pipe, y_gram, A_graft_gram);

    train_func_ngram_lr = lambda model, A, y, idx, validation_idx: train_ngram(model, A, y, idx, None, names, ngram_transform_pipe);
    predict_func_ngram_lr = lambda model, A, idx: predict_ngram(model, A, idx, ngram_transform_pipe, y_gram, A_graft_gram);

    train_func_recurr = lambda model, db_time, y, idx, validation_idx: train_raw(model, db_time, y, idx, validation_idx, names, feature_counter, feature_names);
    predict_func_recurr = lambda model, db_time, idx: predict_raw(model, db_time, idx, names);

    #generate the data and time
    txt_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    #set the storage directory
    results_dir = results_dir+'auditlog_'+txt_time+'_train_'+str(len(cv_cuckoo[0][0]))
    
    if model in ["ngram", "full"]:       
        one_set(A_gram, y_gram, cv_cuckoo, final_model_ngram, names_gram, feature_names, os.path.join(results_dir,'ngram'), train_func=train_func_ngram, predict_func=predict_func_ngram)
        one_set(A_gram, y_gram, cv_time, final_model_ngram, names_gram, feature_names, os.path.join(results_dir,'ngram-time'), train_func=train_func_ngram, predict_func=predict_func_ngram)
    if model in ["ngram_lr", "full"]:
        one_set(A_gram, y_gram, cv_cuckoo, final_model_ngram_lr, names_gram, feature_names, os.path.join(results_dir,'ngram-lr'), train_func=train_func_ngram_lr, predict_func=predict_func_ngram_lr)
        one_set(A_gram, y_gram, cv_time, final_model_ngram_lr, names_gram, feature_names, os.path.join(results_dir,'ngram-lr-time'), train_func=train_func_ngram_lr, predict_func=predict_func_ngram_lr)
    if model in ["conv", "full"]:   
        one_set(db_time, y, cv_cuckoo, final_model_conv, names, feature_names, os.path.join(results_dir,'conv'), train_func=train_func_recurr, predict_func=predict_func_recurr)
        one_set(db_time, y, cv_time, final_model_conv, names, feature_names, os.path.join(results_dir,'conv-time'), train_func=train_func_recurr, predict_func=predict_func_recurr)
    if model == "rnn":   
        one_set(db_time, y, cv_cuckoo, final_model_rnn, names, feature_names, os.path.join(results_dir,'recurrent'), train_func=train_func_recurr, predict_func=predict_func_recurr)
    
    log.info("Finished all!")
    
def train_ngram(model, A, y, idx, validation_idx, names, ngram_transform_pipe):

    A_idx = ngram_transform_pipe.fit_transform(A[idx,:])

    if validation_idx is not None and model.callbacks is not None:
        A_test = ngram_transform_pipe.transform(A[validation_idx,:])
        model.callbacks[0].set_validation(A_test, y[validation_idx])

    model.fit(A_idx, y[idx])

    return model

def predict_ngram(model, A, idx, ngram_transform_pipe, y=None, A_graft=None):
    
    A_idx = A[idx,:]
    if y is not None and A_graft is not None:
        
        log.info('Performing grafting. Grafting matrix size is {0}x{1}.'.format(A_graft.shape[0], A_graft.shape[1]))
        idx_graft = np.array([i for i in xrange(len(idx)) if y[idx[i]]==1])
        
        #generate the grafting matrix
        cz = 0
        M = []
        while (cz<len(idx_graft)):
            A_curr = A_graft[0:np.min([A_graft.shape[0], len(idx_graft)-cz])]
            M.append(A_curr)
            cz += A_curr.shape[0]
        M = scipy.sparse.vstack(M)
        M = M[:len(idx_graft),:]
        
        A_idx[idx_graft,:] = A_idx[idx_graft,:]+M
    
    A_test = ngram_transform_pipe.transform(A_idx)
    prob = model.predict_proba(A_test)

    return prob

def train_raw(model, A_rocksdb, y, idx, validation_idx, names, rocks_counter, features):
    
    idx_names = [names[i] for i in idx]    

    #get the top features
    log.info('Starting count of most common features.')
    rocks_counter.store_counts(A_rocksdb, idx_names, features, '{WINDOWS_NETWORK|')
    
    log.info('Got {} elements. Most popular 100 are'.format(len(rocks_counter.counts)))
    
    print rocks_counter.features[:100]

    if validation_idx is not None and model.callbacks is not None and model.callbacks[0] is not None:
        validation_idx_names = [names[i] for i in validation_idx]
        print model.callbacks[0]
        model.callbacks[0].set_validation_rocksdb(A_rocksdb, validation_idx_names, y[validation_idx])
 
    model = model.fit_rocksdb(A_rocksdb, y[idx], idx_names, read_doubles = True)
    
    return model
    
def predict_raw(model, A_rocksdb, idx, names):
    
    idx_names = [names[i] for i in idx]
    
    prob = model.predict_proba_rocksdb(A_rocksdb, idx_names, read_doubles = True)
    
    return prob
                
def one_set(A, y, cv, final_model, names, feature_names, results_dir, train_func=None, predict_func=None, baseline=None):
    
    log.info("Starting {} analysis.".format(results_dir))

    #create storage directory
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    fpr_array, tpr_array, thresh_array, oob_estimates = validation.compute_cv(cv, final_model, A, y, train_func, predict_func)
    
    log.info("Building storage record.")
    
    result = validation.create_record(final_model, y, cv, names, fpr_array, tpr_array, thresh_array, oob_estimates)
    
    try:
        #if logistic regression get feature weights
        if 'logitreg' in final_model.named_steps:
            logitreg = final_model.named_steps['logitreg']
            logit_out = {}
            logit_out['lambda'] = (1.0/logitreg.Cs_).tolist();
            logit_out['lambda_best'] = (1.0/logitreg.C_).tolist()[0];
            
            #now get the empty
            valid_idx = final_model.named_steps['empty'].get_important_indicies()
            ordered = zip(valid_idx, logitreg.coef_.ravel())
            ordered = sorted(ordered, key=lambda o: -np.abs(o[1]))
            out_dict = []
            max_value = np.abs(ordered[0][1])
            for idx, value in ordered:
                
                if max_value*1.e-6>np.abs(value):
                    break;
                
                out_dict.append({'name' : feature_names[idx], 'value' : value })
             
            logit_out['type'] = 'LogisticRegressionCV'   
            logit_out['nnz'] = len(out_dict)
            logit_out['weights'] = out_dict   
            logit_out['offset'] = logitreg.intercept_[0]
            
            #store the result
            result['model'] = logit_out
    except:
        tb = traceback.format_exc()
        log.error(tb)
    
    log.info('Created results.')
    
    path = validation.store_record(result, results_dir, 'full_time', False)

    log.info('Stored results to directory %s.' % (str(path)))

    log.info("Finished!")    

def main(argv):
    
    baker.run()

if __name__ == '__main__':
    main(sys.argv[1:])
