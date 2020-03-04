#!/bin/python
import numpy as np
import pandas as pd
import os
from sklearn.svm.classes import SVC
import cPickle
import sys
from sklearn.preprocessing import StandardScaler
import glob

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: {0} vocab_file, file_list".format(sys.argv[0])
        print "vocab_file -- path to the vocabulary file"
        print "file_list -- the list of videos"
        exit(1)

    vocab_file = sys.argv[1]; file_list_dir = sys.argv[2]
    
    listing1 = glob.glob('../asrs/*.ctm')
    vocab = cPickle.load(open(vocab_file,"rb"))
    vocab = dict([(y,x) for x,y in vocab.items()])

    bow_list = []
    conf_seq_list = []

    for i in listing1:
        word_list = []
        conf_list = []
        with open(i, "r") as f:
            for line in f.readlines():
                _, _, _, _, word, conf = line.strip('\n').split()        
                if word != '<#s>':
                    word_list.append(vocab[word])
                    conf_list.append(float(conf))
        bow_list.append(word_list)
        conf_seq_list.append(conf_list)
    asr_idx = list(map(lambda x: x.split('/')[-1].split('.')[0], listing1))

    # make a vector for each video
    def bow2vec(bow_list, conf_seq_list, vocab):
        data = np.zeros((len(bow_list), len(vocab)))
        for row, bow in enumerate(bow_list):
            for idx, col in enumerate(bow):
                data[row, col] += conf_seq_list[row][idx]
        return data
    data = bow2vec(bow_list, conf_seq_list, vocab)
    
    # make asr feature DataFrame
    df = pd.DataFrame(index=asr_idx, data=data)
    if not os.path.isdir('asrfeat'):
        os.mkdir('asrfeat')
    #df.to_csv('asrfeat/asr.50.feature.csv')
    # assign all zero for the file that is not able to be matched with asr file
    with open(file_list_dir, "r") as f:
        file_list = f.read().split()
    asr_df = pd.DataFrame(data = np.zeros((len(file_list), len(vocab))), index = file_list)
    asr_df.loc[df.index] = df.values
    asr_df.to_csv('asrfeat/asr.50.feature.csv')
    

    
    print "ASR features generated successfully!"
