# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:45:40 2019

@author: Soriba
"""
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed


class RandomForestClassifierCustom(BaseEstimator):
    """
    Random forest classifier. Uses sklearn's Decision Trees as estimators
    
    Parameters
    ----------
    n_estimators : integer
        The number of trees in the forest
        
    max_depth : integer
        The maximum depth of each tree
        
    max_features : int
        the number of features to consider when looking for the best split
    
    random_state : int
        seed used by the random number generator    
    """
    def __init__(self, criterion='gini', n_estimators=100, max_depth=None,
                 max_features='auto', random_state=None, n_jobs = 1):
        self.criterion = criterion
        self.n_estimators = n_estimators 
        self.max_depth = max_depth 
        self.max_features = max_features 
        self.random_state = random_state
        self.trees = [] #we store the trees in a list
        self.feat_ids_by_tree = [] #for each tree, we store indices of features
        self.n_jobs = n_jobs
        self.n_features = None
        self.oob_samples = []
        
    
    def fit(self, X, y):
        """
        Build a forest
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples
            
        y : array-like of shape (n_samples,)
        
        Returns
        -------
        self : object
        """
        self.n_features = X.shape[1]
        if self.max_features=='auto':
            self.max_features = int(np.sqrt(self.n_features))
            
        for k in range(self.n_estimators):
            #random set of features without replacement
            ids=np.random.choice([k for k in range(self.n_features)],
                           self.max_features,replace=False) 
            #for this tree, the ids of the features that we randomly select
            self.feat_ids_by_tree.append(ids) 
            #random sample with replacement to create a bootstrap sample
            indices=np.random.choice([k for k in range(len(X))],len(X)
                ,replace=True) 
            #we create a bootstrap sample of X
            bootstrap_sample_X=X[indices,:] 
            #we take the indices accordingly for the target
            bootstrap_sample_y=y[indices] 
            oob_idx=np.setdiff1d(range(len(X)),indices)
            oob_sample_X=X[oob_idx,:]
            oob_sample_y=y[oob_idx]
            self.oob_samples.append((oob_sample_X,oob_sample_y))
            tree=DecisionTreeClassifier(max_depth=self.max_depth,
                                       max_features=self.max_features,
                                       random_state=self.random_state)
            #we train a Decision Tree on the bootstrap sample and store it 
            self.trees.append(tree.fit(bootstrap_sample_X[:,ids],
                                      bootstrap_sample_y))
            
        return self #we return the Random Forest
    
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples 
         
        Returns
        -------
        p : array of shape = (n_samples, n_classes)
            The class probabilities of the input samples
        """
        probas=[]
        #for each tree, we predict the probabilities vector
        for k in range(len(self.trees)): 
            ids=self.feat_ids_by_tree[k]
            tree=self.trees[k]
            probas.append(tree.predict_proba(X[:,ids]))
        #we average the probabilities returned by all the trees, 
        #finally we have a probability vector
        return np.mean(probas,axis=0) 
    
    
    def predict(self, X):
        """
        Predict classes for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples 
            
        Returns
        -------
        c : array of shape = (n_samples,)
            The classes of the input samples
        """
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)

        
    def compute_oob_error(self, oob_sample_X, oob_sample_y):
        y_true = oob_sample_y
        y_pred = self.predict(oob_sample_X)
        return 1-accuracy_score(y_true, y_pred)
        
        
    def mean_decrease_accuracy(self):
        """
        Return the feature importances using mean decrease accuracy (not 
        implemented by sklearn)
        """
        all_importances = np.zeros(self.n_features)
        for i in range(self.n_features):
            oob_sample_X, oob_sample_y = self.oob_samples[i]
            oob_error = self.compute_oob_error(oob_sample_X, oob_sample_y)
            #indices to permute for this feature
            indices_p = list(range(len(oob_sample_X)))
            random.shuffle(indices_p)
            oob_sample_X_p = oob_sample_X.copy() 
            oob_sample_X_p[:,i] = oob_sample_X[indices_p,   i]
            oob_error_p = self.compute_oob_error(oob_sample_X_p, oob_sample_y)
            all_importances[i]=oob_error_p - oob_error
        
        return all_importances/np.sum(all_importances)
            
        
    def feature_importances_(self):
        """
        Return the feature importances (the higher, the more important the
           feature). Calculated using mean decrease impurity
        Returns
        -------
        feature_importances_ : array, shape = [n_features]
            The values of this array sum to 1, unless all trees are single node
            trees consisting of only the root node, in which case it will be an
            array of zeros.
        """
        n_trees = len(self.trees)
        importances_by_tree = Parallel(n_jobs=self.n_jobs,)(
            delayed(getattr)(tree, 'feature_importances_')
            for tree in self.trees if tree.tree_.node_count > 1)
        
        all_importances = []

        if not importances_by_tree:
            return np.zeros(self.n_features_, dtype=np.float64)
        
        for tree_id in range(n_trees):
            ids = self.feat_ids_by_tree[tree_id]
            importances = np.zeros(self.n_features)
            importances.fill(np.nan)
            importances[ids] = importances_by_tree[tree_id]
            all_importances.append(importances)        
        all_importances = np.nanmean(all_importances,
                                  axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)