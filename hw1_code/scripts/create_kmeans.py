#!/bin/python
import numpy as np
import pandas as pd
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
from collections import Counter
import sys
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)

    kmeans_model = sys.argv[1]; file_list_dir = sys.argv[3]
    cluster_num = int(sys.argv[2])
  
    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model,"rb"))
    print "creating K-means features.."
    
    def get_bag_of_words(mfcc, kmeans):
        pred = kmeans.predict(mfcc)    
        bow_vector_count = dict(Counter(pred))
        bow_dummy = np.zeros((1, cluster_num))
        bow_dummy[0, np.array(bow_vector_count.keys())-1] = bow_vector_count.values()
        return bow_dummy
    
    # open HVCXXXX.mfcc.csv
    with open(file_list_dir, "r") as f:
        file_list = f.read().split()
    bows  = []
    for file_name in file_list:
        mfcc_dir = 'mfcc/{}.mfcc.csv'.format(file_name)
        try:
            mfcc = pd.read_csv(mfcc_dir, delimiter=';', header=None)
            bow = get_bag_of_words(mfcc, kmeans)
            bows.append(bow)
        # assign zeros vector in the case of no audio file
        except:
            bow = np.zeros((1, cluster_num))
            bows.append(bow)
        
    # make DataFrame file_name, BOW
    df = pd.DataFrame(data=np.vstack(bows))
    df.index = file_list
    
    # save kmeans feature as csv files
    #feature_dir = '/home/ubuntu/11775-hws/hw1_code/feature/'
    if not os.path.isdir('kmeans/'):
        os.mkdir('kmeans/')
    df.to_csv('kmeans/kmeans.{}.feature.csv'.format(cluster_num))    
    print "K-means features generated successfully!"
