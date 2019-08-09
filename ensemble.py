# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:45:40 2019

@author: Soriba
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator


class RandomForestClassifierCustom(BaseEstimator):
    """
    Random forest classifier
    
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
    def __init__(self, n_estimators=100, max_depth=10, max_features=10, 
                 random_state=75):
        self.n_estimators = n_estimators 
        self.max_depth = max_depth 
        self.max_features = max_features 
        self.random_state = random_state
        self.trees = [] #we store the trees in a list
        self.feat_ids_by_tree = [] #for each tree, we store indices of random set of features
        
    
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
        for k in range(self.n_estimators):
            ids=np.random.choice([k for k in range(X.shape[1])],
                           self.max_features,replace=False) #random sample of features without replacement
            self.feat_ids_by_tree.append(ids) #for this tree, the ids of the features that we randomly select
            indices=np.random.choice([k for k in range(len(X))],len(X)
                ,replace=True) #random sample with replacement to create a bootstrap sample
            bootstrap_sample_X=X[indices,:] #we create a bootstrap sample of X
            bootstrap_sample_y=y[indices] #we take the indices accordingly for the target vector
            tree=DecisionTreeClassifier(max_depth=self.max_depth,
                                       max_features=self.max_features,
                                       random_state=self.random_state)
            self.trees.append(tree.fit(bootstrap_sample_X[:,ids],
                                      bootstrap_sample_y))
            #we train a Decision Tree on the bootstrap sample and store it in the list of the trees
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
        for k in range(len(self.trees)): #for each tree, we predict the probabilities vector
            ids=self.feat_ids_by_tree[k]
            tree=self.trees[k]
            probas.append(tree.predict_proba(X[:,ids]))
        return np.mean(probas,axis=0) #we average the probabilities returned by all the trees, finally we have a probability vector
    
    
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
        