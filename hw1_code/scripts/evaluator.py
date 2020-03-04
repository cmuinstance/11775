#!/bin/python2.5

import sys
import os
from sklearn.metrics import average_precision_score

if __name__=="__main__":
    #   load the ground-truth file list
    y_true_dir = sys.argv[1]
    y_pred_dir = sys.argv[2]
    
    event = y_pred_dir.split('/')[1].split('_')[0]
    
    with open(y_true_dir, 'r') as f:
        y_true = [line.split()[1] for line in f.readlines()]
    y_true = list(map(lambda x: float(x==event), y_true))
    #print y_true[:10]
    with open(y_pred_dir, 'r') as f:
        y_pred = f.read().split()
    y_pred = list(map(lambda x: float(x), y_pred))
    print "Average precision: ",average_precision_score(y_true,y_pred)
            