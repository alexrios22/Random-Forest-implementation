# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:45:40 2019

@author: Soriba
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=10, max_features=10, 
                 random_state=75):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feat_ids_by_tree = []
        
    def fit(self, X, y):
        for k in range(self.n_estimators):
            np.random.seed(self.random_state+k)
            ids=np.random.choice([k for k in range(X.shape[1])],
                           self.max_features,replace=False)
            self.feat_ids_by_tree.append(ids)
            indices=np.random.choice([k for k in range(len(X))],len(X)
                ,replace=True)
            bootstrap_sample_X=X[indices,:]
            bootstrap_sample_y=y[indices]
            clf=DecisionTreeClassifier(max_depth=self.max_depth,
                                       max_features=self.max_features,
                                       random_state=self.random_state)
            self.trees.append(clf.fit(bootstrap_sample_X[:,ids],
                                      bootstrap_sample_y))
        return self
            
            
    def predict_proba(self, X):
        probas=[]
        for k in range(len(self.trees)):
            ids=self.feat_ids_by_tree[k]
            tree=self.trees[k]
            probas.append(tree.predict_proba(X[:,ids]))
        return np.mean(probas,axis=0)