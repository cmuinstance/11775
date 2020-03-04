#!/bin/python 

import numpy as np
import pandas as pd
import os
from sklearn.svm.classes import SVC
import cPickle
import sys
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the svm model"
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    #output_file = sys.argv[4]
    save_dir = sys.argv[4]
    
    if not os.path.isdir(feat_dir):
        os.mkdir(feat_dir)
    if feat_dir == 'kmeans/':
        feat_dir = feat_dir + 'kmeans.50.feature.csv'
    elif feat_dir == 'asrfeat/':
        feat_dir = feat_dir + 'asr.50.feature.csv'
    output_file = 'svm.{}.model'.format(event_name)
    
    # get DataFrame for feature
    df = pd.read_csv(feat_dir, index_col=0)
    def get_svm_input_output(input_list_dir, output_list_dir):
        with open(input_list_dir, 'r') as f:
            input_list = f.read().split()
        # get intersection between feature index and training index
        #input_list = list(set(df.index).intersection(set(input_list)))
        #print 'model will be trained on {} data (may be less than the number of whole train set)'.format(len(input_list))
        print 'Training model on {} data ...'.format(len(input_list))
        scaler = StandardScaler()
        X = df.loc[input_list].values
        X = X/(X.sum(axis=1).reshape(-1,1) + 1e-8)
        X = scaler.fit_transform(X)

        with open(output_list_dir, 'r') as f:
            y = [line.split()[1] for line in f.readlines()]
        y = list(map(lambda x: int(x == event_name), y))
        return X, y
        
        #d = dict(x)
        #y1 = list(map(lambda x: d[x], input_list))
        #y1 = list(map(lambda x: int(x == event_name), y1))
        #return X, y1

    X, y = get_svm_input_output(input_list_dir = 'list/train.video', output_list_dir = '../all_trn.lst')
    if event_name == 'P001':
        svc = SVC(C=100, gamma=0.1, class_weight={0:1,1:20}, probability=True)
    if event_name == 'P002':
        svc = SVC(C=100, gamma=0.005, class_weight={0:1,1:20}, probability=True)
    if event_name == 'P003':
        svc = SVC(C=100, gamma=0.0007, class_weight={0:1,1:20}, probability=True)
    svc = svc.fit(X, y)

    # save svc model
    #save_dir = 'mfcc_pred/'
    if not os.path.isdir('mfcc_pred/'):
        os.mkdir('mfcc_pred/')
    cPickle.dump(svc, open(save_dir, "wb"))
    print 'SVM trained successfully for event %s!' % (event_name)
