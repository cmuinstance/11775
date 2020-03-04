#!/bin/python 

import numpy as np
import pandas as pd
import os
from sklearn.svm.classes import SVC
import cPickle
import sys
from sklearn.preprocessing import StandardScaler


# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0])
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features; provided just for debugging"
        print "output_file -- path to save the prediction score"
        exit(1)

    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    
      
    # load SVM model
    svc = cPickle.load(open(model_file,"rb"))
    # load feature
    if feat_dir == 'kmeans/':    
        df = pd.read_csv(feat_dir + 'kmeans.50.feature.csv', index_col=0)
    elif feat_dir == 'asrfeat/':
        df = pd.read_csv(feat_dir + 'asr.50.feature.csv', index_col=0)
    else:
        raise ValueError('Error! check feat_dir')
    
    
#     # ---------------------Validation
#     # read validation video list
#     list_dir = 'list/val.video'
#     # apply svm model to test_lst
#     with open(list_dir, 'r') as f:
#         test_list = f.read().split()
#     #test_list = list(set(df.index).intersection(set(test_list)))
        
#     # normalize dataset 
#     scaler = StandardScaler()
#     X = df.loc[test_list].values
#     X = X/(X.sum(axis=1).reshape(-1,1) + 1e-8)
#     X = scaler.fit_transform(X)
    
#     # apply svm model to test set
#     decision_score = svc.predict_proba(X)[:,1]
#     #decision_score = svc.predict(X).flatten()
#     with open(output_file, 'w') as f:
#         for idx, score in enumerate(decision_score):
#             #f.write(test_list[idx] + ' ' + str(round(score, 3)))
#             f.write(str(round(score, 3)))
#             f.write('\n')
    
    # ---------------------Test
    # read test video list
    test_list_dir = '../all_test.video'
    # apply svm model to test_lst
    with open(test_list_dir, 'r') as f:
        test_list = f.read().split()

    scaler = StandardScaler()
    X = df.loc[test_list].values
    X = X/(X.sum(axis=1).reshape(-1,1) + 1e-8)
    X = scaler.fit_transform(X)
    
    # apply svm model to test set
    decision_score = svc.predict_proba(X)[:,1]
    #decision_score = svc.predict(X).flatten()
    with open(output_file, 'w') as f:
        for idx, score in enumerate(decision_score):
            f.write(test_list[idx] + ' ' + str(round(score, 3)))
            #f.write(str(round(score, 3)))
            f.write('\n')