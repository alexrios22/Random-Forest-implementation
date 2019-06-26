# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:45:40 2019

@author: Soriba
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator

class RandomForestClassifierCustom(BaseEstimator):
    def __init__(self, n_estimators=10, max_depth=10, max_features=10, 
                 random_state=75):
        self.n_estimators = n_estimators #number of trees in the forest
        self.max_depth = max_depth #maximum depth of each tree
        self.max_features = max_features #the number of features used to make the splits
        self.random_state = random_state
        self.trees = [] #we store the trees in a list
        self.feat_ids_by_tree = [] #for each tree, random set of features
        
    def fit(self, X, y):
        for k in range(self.n_estimators):
            np.random.seed(self.random_state+k) #for the reproducibility of the results, while keeping a certain 'randomness'
            ids=np.random.choice([k for k in range(X.shape[1])],
                           self.max_features,replace=False) #random sample without replacement
            self.feat_ids_by_tree.append(ids) #for this tree, the ids of the features that we randomly select
            indices=np.random.choice([k for k in range(len(X))],len(X)
                ,replace=True) #random sample with replacement to create a bootstrap sample
            bootstrap_sample_X=X[indices,:] #we create a bootstrap sample of X
            bootstrap_sample_y=y[indices] #we take the indices accordingly for the target vector
            clf=DecisionTreeClassifier(max_depth=self.max_depth,
                                       max_features=self.max_features,
                                       random_state=self.random_state)
            self.trees.append(clf.fit(bootstrap_sample_X[:,ids],
                                      bootstrap_sample_y))
            #we train a Decision Tree on the bootstrap sample and store it in a list
        return self #we return the Random Forest
            
            
    def predict_proba(self, X):
        probas=[]
        for k in range(len(self.trees)): #for each tree, we predict the probabilities vector
            ids=self.feat_ids_by_tree[k]
            tree=self.trees[k]
            probas.append(tree.predict_proba(X[:,ids]))
        return np.mean(probas,axis=0) #we average the probabilities returned by all the trees, finally we have a probability vector