from __future__ import division
import matplotlib
from __builtin__ import dict
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import logging
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, classification_report
import datetime
from scipy import integrate
import json
import gzip
import io
from bisect import bisect_left
from basicutils import one_hot_to_label

def FindLastLessThanOrEqual(a, x):
    hi = len(a)
    pos = bisect_left(a,x,0,hi)
    return (pos if pos != hi else -1)

def stepinterp(fpr_desired, fpr, tpr):
    
    tpr_desired = np.zeros(len(fpr_desired))
    for pos, val in enumerate(fpr_desired):
        idx = FindLastLessThanOrEqual(fpr, val)
        if idx >= 0:
            tpr_desired[pos] = tpr[idx]
            
    return tpr_desired    

def train_model(model, rocksdb, y, idx, idx_test=None):
    
    rocksdb = rocksdb[idx,:]
    y = y[idx]

    model = model.fit(rocksdb, y)
    return model


def predict_model(model, A,  idx, train_idx=None):
    
    A = A[idx,:]

    pred = model.predict(A)
    
    return pred

def store_record(record, path, name, add_time = True):
    
    if add_time:
        path = os.path.join(path, name+'_'+record['time'])
    else:
        path = os.path.join(path, name)
    
    if not os.path.exists(path):
        os.makedirs(path)
 
    store = record['output']
    validation = record['validation']
    
    with open(os.path.join(path, 'output.json'), 'w') as f:
        json.dump(store, f, indent=4, separators=(',', ': '), sort_keys=True)
    
    with gzip.GzipFile(os.path.join(path, 'validation.json.gz'), 'w') as f:
        json.dump(validation, f, indent=4, separators=(',', ': '), sort_keys=True)
    
    if 'model' in record:
        with gzip.GzipFile(os.path.join(path, 'model.json.gz'), 'w') as f:
            json.dump(record['model'], f, indent=4, separators=(',', ': '), sort_keys=True)

    for key, value in record['images'].iteritems():
            with open(os.path.join(path, key+'.png'), 'w') as f:
                value.seek(0)
                f.write(value.read())
    
    return path


def create_record(model, y, cv, names, fpr_array, tpr_array, thresh_array, oob_estimates, baseline=None):
    
    txt_time = str(datetime.datetime.now())

    final_prediction = np.matrix([[float(y[i]), float(p)] for i,p in oob_estimates.iteritems()]);
    fpr, tpr, thresh = roc_curve(final_prediction[:,0], final_prediction[:,1], 1)
    curr_auc = auc(fpr, tpr)

    #compute the shading over a large number of points
    sp = 1000;
    fpr_points = np.concatenate([fpr, np.logspace(-6,-5, sp), np.linspace(1e-5, 1e-4, sp), np.linspace(1e-4, 1e-3, sp), np.linspace(1e-3, 1e-2, sp), np.linspace(1e-2, 1, sp)])
    fpr_points = np.sort(fpr_points)
    mean_fpr, mean_tpr, std_tpr = compute_stat_cv(fpr_array, tpr_array, fpr_points)

    #get the index
    idx_1e2 = (np.abs(fpr-1e-2)).argmin()
    idx_1e3 = (np.abs(fpr-1e-3)).argmin()
    idx_1e4 = (np.abs(fpr-1e-4)).argmin()

    #get the values
    auc_1e2 = integrate.trapz(tpr[:idx_1e2], fpr[:idx_1e2])*1e2
    auc_1e3 = integrate.trapz(tpr[:idx_1e3], fpr[:idx_1e3])*1e3
    auc_1e4 = integrate.trapz(tpr[:idx_1e4], fpr[:idx_1e4])*1e4
    
    #plt.semilogx(mean_fpr, mean_tpr, 'k-',  label='Mean ROC (area = %0.3f, tpr = %0.3f)' % (mean_auc, mean_tpr[idx_1e3]))
    #plt.xlim([1.0e-4, 1.0])
    
    if baseline is None:
        plt.plot(np.logspace(-10,0, 1000), np.logspace(-10,0, 1000), 'k--')
    else:
        plt.plot(baseline[0], baseline[1],'k--')
    
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=.4, label='95% Confidence Interval')
    
    plt.step(fpr, tpr, 'k-',  label='ROC (AUC = %0.6f, AUC_1e-3, = %0.6f, TPR_1e-4 = %0.6f, TPR_1e-3 = %0.6f, )' % (curr_auc, auc_1e3, tpr[idx_1e4], tpr[idx_1e3]))
    
    cv_full = [[train,test] for (train,test) in cv]
    
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC, Training: %d, Testing: %d, cv: %d' % (len(cv_full[0][0]), len(cv_full[0][1]), len(cv_full)))
    plt.legend(loc="lower right", prop={'size':8})
    plt.tight_layout()
    #plt.show()
        
    saved_params = {}
    try:
        params = model.get_params()
        for k in params:
            if type(params[k]) in (str,int,np.array,list,dict):
                try:
                    json.dumps(params[k])                
                    saved_params[k] = params[k]
                except TypeError:
                    saved_params[k] = repr(params[k])
    except:
        pass    
    
    #store the output
    store = dict()
    store['params'] = saved_params
    store['roc'] = np.column_stack((fpr, tpr, thresh)).tolist()
    #store['std_tpr'] = std_tpr.tolist()
    store['auc'] = float(curr_auc)
    store['tpr_1e2'] = float(mean_tpr[idx_1e2])
    store['auc_1e2'] = float(auc_1e2)
    store['tpr_1e3'] = float(mean_tpr[idx_1e3])
    store['auc_1e3'] = float(auc_1e3)
    store['tpr_1e4'] = float(mean_tpr[idx_1e4])
    store['auc_1e4'] = float(auc_1e4)
    store['cv_size'] = len(cv_full)
    store['size'] = len(cv_full[0][0])+len(cv_full[0][1])
    store['benign_size'] = int((y==0).sum())
    store['malware_size'] = int((y==1).sum())
    store['cv_train'] = len(cv_full[0][0])
    store['cv_test'] = len(cv_full[0][1])
    
    pos = [{'name': names[i], 'p' : float(p), 'label' : float(y[i])} for i,p in oob_estimates.iteritems() if y[i]==1];
    neg = [{'name': names[i], 'p' : float(p), 'label' : float(y[i])} for i,p in oob_estimates.iteritems() if y[i]==0];
    
    pos = sorted(pos, key=lambda v: v['p'])
    neg = sorted(neg, key=lambda v: -v['p'])
    
    store['top_fp'] = neg[0:500];    
    store['top_fn'] = pos[0:500];
    
    val_results = {}
    val_results['oob_estimates'] = list(np.concatenate((neg, pos)))
            
    #save the model
    img_results = {}
    
    buf = io.BytesIO()    
    plt.xlim([0, 0.0001])
    plt.savefig(buf)
    img_results['img_0_0001'] = buf
    
    buf = io.BytesIO()    
    plt.xlim([0, 0.001])
    plt.savefig(buf)
    img_results['img_0_001'] = buf

    buf = io.BytesIO()    
    plt.xlim([0, 0.01])
    plt.savefig(buf)
    img_results['img_0_01'] = buf
   
    buf = io.BytesIO()    
    plt.xlim([0, 0.1])
    plt.savefig(buf)
    img_results['img_0_1'] = buf

    buf = io.BytesIO()    
    plt.xlim([0, 1])
    plt.savefig(buf)
    img_results['img_1'] = buf
    
    buf = io.BytesIO()    
    plt.xlim([1e-6, 1])
    plt.xscale('log')
    plt.savefig(buf)
    img_results['img_log'] = buf

    results = {}
    results['time'] = txt_time
    results['output'] = store
    results['validation'] = val_results
    results['images'] = img_results
    
    plt.close()

    return results

def create_record_regression(model, y, cv, names, loss_array, oob_estimates):
        
    txt_time = str(datetime.datetime.now())
    
    loss_values = oob_estimates.values()
    loss = np.mean(loss_values)
    
    plt.hist(loss_values, 200, weights=np.ones(len(loss_values))/len(loss_values), label='Loss Distribution: loss={}'.format(loss))
    
    cv_full = [[train,test] for (train,test) in cv]
    
    plt.xlabel('Loss')
    plt.ylabel('Fraction')
    plt.title('Loss, Training: %d, Testing: %d, cv: %d' % (len(cv_full[0][0]), len(cv_full[0][1]), len(cv_full)))
    plt.legend(loc="upper center")
    plt.tight_layout()
    #plt.show()
        
    saved_params = {}
    try:
        params = model.get_params()
        for k in params:
            if type(params[k]) in (str,int,np.array,list,dict):
                try:
                    json.dumps(params[k])                
                    saved_params[k] = params[k]
                except TypeError:
                    saved_params[k] = repr(params[k])
    except:
        pass    
    
    #store the output
    store = dict()
    store['params'] = saved_params
    store['loss'] = float(loss)
    store['cv_size'] = len(cv_full)
    store['size'] = len(cv_full[0][0])+len(cv_full[0][1])
    store['benign_size'] = float((y==0).sum())
    store['malware_size'] = float((y==1).sum())
    store['cv_train'] = len(cv_full[0][0])
    store['cv_test'] = len(cv_full[0][1])
    
    loss_store = [{'name': names[i], 'loss' : float(p)} for i,p in oob_estimates.iteritems()];    
    loss_store = sorted(loss_store, key=lambda v: -v['loss'])
    
    store['loss_top'] = loss_store[0:500];    
    store['loss_bottom'] = loss_store[-500:-1];
    
    val_results = {}
    val_results['oob_estimates'] = loss_store
            
    #save the model
    img_results = {}
    
    buf = io.BytesIO()    
    plt.savefig(buf)
    img_results['img_loss'] = buf
    
    results = {}
    results['time'] = txt_time
    results['output'] = store
    results['validation'] = val_results
    results['images'] = img_results
    
    plt.close()

    return results

def create_record_categorical(model, y, cv, names, class_names, conf_matrix_list, oob_estimates, baseline=None):
    
    txt_time = str(datetime.datetime.now())

    saved_params = {}
    try:
        params = model.get_params()
        for k in params:
            if type(params[k]) in (str,int,np.array,list,dict):
                try:
                    json.dumps(params[k])                
                    saved_params[k] = params[k]
                except TypeError:
                    saved_params[k] = repr(params[k])
    except:
        pass    

    cv_full = [[train,test] for (train,test) in cv]
    y_label = one_hot_to_label(y)
    
    samples = [{'name': names[i], 'prob_label' : float(p[y_label[i]]), 'prob_pred' : float(p[np.argmax(p)]), 'pred' : float(np.argmax(p)), 'label' : float(y_label[i]), 'class_label' : class_names[y_label[i]], 'class_pred' : class_names[np.argmax(p)]} for i,p in oob_estimates.iteritems()];
    samples = sorted(samples, key=lambda v:  v['prob_label'])
    samples_predicted = sorted(samples, key=lambda v:  -v['prob_label'])
        
    #get the classification report
    y_pred = []
    y_label = []
    for sample in samples:
        y_pred.append(sample['pred'])
        y_label.append(sample['label'])
    y_pred = np.array(y_pred)
    y_label = np.array(y_label)
    
    weighted_f1 = f1_score(y_label, y_pred, average='weighted')  
    
    conf_matrix = confusion_matrix(y_label, y_pred, labels=range(len(class_names)))
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    cmap_viridis =  matplotlib.cm.get_cmap('viridis')
    fig_size = plt.gcf().get_size_inches()
    plt.figure(figsize=fig_size*3)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap_viridis)
    plt.title('CM, Train: %d, Test: %d, cv: %d, F1: %0.4f' % (len(cv_full[0][0]), len(cv_full[0][1]), len(cv_full), weighted_f1))
    plt.clim(0,1)
    plt.colorbar()
    
    if len(class_names)<200:
        tick_marks = np.arange(len(class_names))    
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)
        plt.grid(True)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    #store the output
    store = dict()
    store['params'] = saved_params
    store['weighted_f1'] = weighted_f1
    store['size'] = len(cv_full[0][0])+len(cv_full[0][1])
    store['cv_size'] = len(cv_full)
    store['cv_train'] = len(cv_full[0][0])
    store['cv_test'] = len(cv_full[0][1])
            
    store['report'] = classification_report(y_label, y_pred, labels=range(len(class_names)), target_names=class_names, digits=3)
    store['top_missed'] = samples[0:200];
    store['top_predicted'] = samples_predicted[0:200];
    
    #print the report
    print store['report']
    
    val_results = {}
    val_results['oob_estimates'] = samples
    val_results['conf_matrix'] = conf_matrix.tolist()
    
    img_results = {}

    buf = io.BytesIO()    
    plt.savefig(buf)
    img_results['confusion_matrix'] = buf
    
    results = {}
    results['time'] = txt_time
    results['output'] = store
    results['validation'] = val_results
    results['images'] = img_results

    plt.close()

    return results

def compute_stat_cv(fpr_array, tpr_array, fpr_points):
    
    tpr_list = []
    for i in xrange(len(tpr_array)):
        tpr_list.append(stepinterp(fpr_points, fpr_array[i], tpr_array[i]))
        
    tpr_matrix = np.vstack(tpr_list)
    
    mean_tpr = np.mean(tpr_matrix,axis=0)
    std_tpr = np.std(tpr_matrix,axis=0)

    return fpr_points, mean_tpr, std_tpr


def compute_cv_categorical(cv, model, A, y, train_func=None, predict_func=None, predict_only=False):
    
    log = logging.getLogger(__name__)

    if train_func is None:
        train_func = train_model
    if predict_func is None:
        predict_func = predict_model
       
    log.info("Starting cross-validation.");
    
    results_dict = {}

    #check the data
    cv_size = 0
    for i, (train, test) in enumerate(cv):
        
        if set(train).intersection(test) :
            log.warn("The training set intersects testing set.")
        log.info("Cross-validation %d has: training size: %d, testing size: %d:" % (i+1, len(train), len(test)));
        cv_size+=1

    #confusion matrix list
    conf_matrix_list = []
    
    for i, (train, test) in enumerate(cv):
        
        log.info("Starting cross-validation %d of %d, training size: %d, testing size: %d" % (i+1, cv_size, len(train), len(test)));

        if not predict_only:
            model = train_func(model, A, y, train, test)
            log.info("Finished training. Predicting test data.");

        #perform the prediction
        y_pred = predict_func(model, A, test)
        
        #get the probability
        y_label = one_hot_to_label(y[test,...])
        y_pred_label = one_hot_to_label(y_pred)

        weighted_f1 = f1_score(y_label, y_pred_label, average='weighted')  
        conf_matrix = confusion_matrix(y_label, y_pred_label)
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        conf_matrix_list.append(conf_matrix)
        
        log.info("Finished predicting testing data. Current iteration estimated weighted f1={0}.".format(weighted_f1));

        #record the data
        for idx,t in enumerate(test):
            if t in results_dict:
                results_dict[t] = np.append(results_dict[t], y_pred[idx,...])
            else:
                results_dict[t] = np.array([y_pred[idx,...]])
                    
    oob_estimates = {}    
    for key, values in results_dict.iteritems():
        oob_estimates[key] = np.mean(np.vstack(values), axis=0)
        
    log.info("Finished CV!");
    
    return conf_matrix_list, oob_estimates


def compute_cv(cv, model, A, y, train_func=None, predict_func=None, predict_only=False):
    
    log = logging.getLogger(__name__)

    if train_func is None:
        train_func = train_model
    if predict_func is None:
        predict_func = predict_model
       
    log.info("Starting cross-validation.");
    
    tpr_array = list();
    fpr_array = list();
    thresh_array = list();
    
    results_dict = {}

    #check the data
    cv_size = 0
    for i, (train, test) in enumerate(cv):
        
        y_train = y[train]
        y_test  = y[test]
        
        if (len(set(y_train))!=2):
            log.warn("Number of label types not equal 2 in training set. Actual = {0}.".format(len(set(y_train))))
        assert len(set(y_test))==2, "Number of label types not equal 2 in test set. Actual = {0}.".format(len(set(y_test)))
        if set(train).intersection(test) :
            log.warn("The training set intersects testing set.")
        log.info("Cross-validation %d has: training size: %d, testing size: %d:" % (i+1, len(train), len(test)));
        log.info("Training set: negative = %d, positive = %d." % (sum(y_train==0), sum(y_train==1)));
        log.info("Testing set: negative = %d, positive = %d." % (sum(y_test==0), sum(y_test==1)));
        
        cv_size+=1

    for i, (train, test) in enumerate(cv):
        
        log.info("Starting cross-validation %d of %d, training size: %d, testing size: %d" % (i+1, cv_size, len(train), len(test)));

        if not predict_only:
            model = train_func(model, A, y, train, test)
            log.info("Finished training. Predicting test data.");

        #get the probability
        preds = predict_func(model, A, test)
        
        if len(preds.shape)==1:
            preds = np.array([preds]).T
        
        if preds.shape[1]==1:
            p_neg = 1-preds
            preds = np.hstack((p_neg, preds))
            
        #convert to format
        preds = preds.astype(np.float)
        
        #get roc curve
        fpr, tpr, thresh = roc_curve(y[test], preds[:,1], 1)        
        curr_auc = auc(fpr, tpr)

        log.info("Finished predicting testing data. Current iteration estimated AUC={0}.".format(curr_auc));

        fpr_array.append(fpr)
        tpr_array.append(tpr)
        thresh_array.append(thresh)

        #record the data
        for idx,t in enumerate(test):
            if t in results_dict:
                results_dict[t] = np.append(results_dict[t], preds[idx,1])
            else:
                results_dict[t] = np.array([preds[idx,1]])
                    
    oob_estimates = {}    
    for key, values in results_dict.iteritems():
        oob_estimates[key] = np.mean(values)
        
    log.info("Finished CV!");
    
    return fpr_array, tpr_array, thresh_array, oob_estimates

def compute_cv_regression(cv, model, A, y, loss_func, train_func=None, predict_func=None, predict_only=False):
    
    log = logging.getLogger(__name__)

    if train_func is None:
        train_func = train_model
    if predict_func is None:
        predict_func = predict_model
       
    log.info("Starting cross-validation.");
    
    loss_array = list();
    
    results_dict = {}

    #check the data
    cv_size = 0
    for i, (train, test) in enumerate(cv):
        
        if set(train).intersection(test) :
            log.warn("The training set intersects testing set.")
        log.info("Cross-validation %d has: training size: %d, testing size: %d:" % (i+1, len(train), len(test)));
        
        cv_size+=1

    for i, (train, test) in enumerate(cv):
        
        log.info("Starting cross-validation %d of %d, training size: %d, testing size: %d" % (i+1, cv_size, len(train), len(test)));

        if not predict_only:
            model = train_func(model, A, y, train, test)
            log.info("Finished training. Predicting test data.");

        #get the value
        pred = predict_func(model, A, test)
        
        #compute individual
        curr_loss = []
        for i,v in enumerate(test):
            
            l = loss_func(np.ravel(y[v,...]).astype(np.float64), np.ravel(pred[i,...]).astype(np.float64))
            curr_loss.append(l)
            
        curr_loss = np.array(curr_loss)
        loss = np.mean(curr_loss)

        log.info("Finished predicting testing data. Current iteration estimated loss={}.".format(loss));

        loss_array.append(curr_loss)

        #record the data
        for idx,t in enumerate(test):
            if t in results_dict:
                results_dict[t] = np.append(results_dict[t], curr_loss[idx])
            else:
                results_dict[t] = np.array([curr_loss[idx]])
                
    oob_estimates = {}    
    for key, values in results_dict.iteritems():
        oob_estimates[key] = np.mean(values)
        
    log.info("Finished CV!");
    
    return loss_array, oob_estimates

