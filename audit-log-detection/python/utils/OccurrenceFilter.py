import logging
import numpy as np
import scipy
import itertools
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.sparsefuncs import mean_variance_axis, count_nonzero
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction import FeatureHasher

class HashTrick(BaseEstimator, TransformerMixin):
    
    log = logging.getLogger('invincea')
    
    def __init__(self, *args, **kwargs):

        self.hasher = FeatureHasher(*args, **kwargs)   

    def fit(self, X=None, y=None):
        
        X = self.data_transform_(X)
        self.hasher.fit(X, y) 
        
        return self 

    def transform(self, X, y=None):
        
        X = self.data_transform_(X)
        
        return self.hasher.transform(X, y) 
    
    def data_transform_(self, X):
        
        if X is None:
            return X
        
        vect = []
        for i in xrange(X.shape[0]):
            vect.append({})
        
        if scipy.sparse.issparse(X):
            cx = X.tocoo()    
            for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
                vect[i][str(j)] = v
        else:
            
            for i in xrange(X.shape[0]):
                for j in xrange(X.shape[1]):
                    vect[i][str(j)] = X[i][j]
            

        X = vect
            
        return X
    
    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.
        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.
        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.
        y : numpy array of shape [n_samples]
            Target values.
        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.transform(X)
        
    
class LogScale(BaseEstimator, SelectorMixin):
    
    log = logging.getLogger('invincea')

    def __init__(self):
        pass

    def fit(self, X, y=None):
        
        self.counts_ = np.ones((1,X.shape[1]))
        
        return self
    
    def transform(self, X):
        
        if scipy.sparse.issparse(X):
            X = X.todense()
        X_new = np.log(X+1.0)
        
        return X_new
    
    def _get_support_mask(self):
        check_is_fitted(self, 'counts_')

        return self.counts_


class CountThreshold(BaseEstimator, SelectorMixin):
    """Feature selector that removes all features below an occurrence threshold.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Parameters
    ----------
    threshold : int
        Features that have less non-zero entries than this value will be removed

    Attributes
    ----------
    counts_ : array, shape (n_features,)
        Counts of non-zero entries for individual features.

    Examples
    --------
    The following dataset has integer features, two of which are the same
    in every sample. These are removed with the default setting for threshold::

        >>> X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
        >>> selector = VarianceThreshold()
        >>> selector.fit_transform(X)
        array([[2, 0],
               [1, 4],
               [1, 1]])
    """

    log = logging.getLogger('invincea')

    def __init__(self, count=2):
        self.count = count
        self.num_columns_ = 0

    def fit(self, X, y=None):
        """Get counts of non-zero entries for columns in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors from which to compute counts.

        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self
        """
        #CountThreshold.log.info("Starting the frequency transform fit for %d dimension matrix." % (X.shape[1]))

        X = check_array(X, ('csr'), dtype=np.float64)

        if hasattr(X, "toarray"):   # sparse matrix
            self.counts_ = count_nonzero(X, axis=0)
        else:
            self.counts_ = np.sum(X > 0, axis=0)
            
        #recompute the threshold
        if self.count<1.0:
            self.threshold = X.shape[0]*self.count;
        else:
            self.threshold = self.count;    
            
        if np.all(self.counts_ <= self.threshold):
            msg = "No feature in X meets the count threshold {0}"
            if X.shape[0] == 1:
                msg += " (X contains only one sample)"
            raise ValueError(msg.format(self.threshold))
        
        self.num_columns_ = X.shape[1]
        
        return self
    
    def get_important_indicies(self):
        
        idx = np.array(range(self.num_columns_))+1
        A = np.matrix(idx)
        v = self.transform(A)
        
        return np.squeeze(np.asarray(v)-1)
    
    def transform(self, X):
        
        Z = super(CountThreshold, self).transform(X)
        
        #CountThreshold.log.info("Reduced %d by %d feature matrix to %d by %d dimension matrix." % (X.shape[0], X.shape[1], Z.shape[0], Z.shape[1]))
        
        return Z

    def _get_support_mask(self):
        check_is_fitted(self, 'counts_')

        return self.counts_ >= self.threshold
