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
    vocab_list = set()
    listing1 = glob.glob('../asrs/*.ctm')
    for i in listing1:
        with open(i, "r") as f:
            for line in f.readlines():
                _, _, _, _, word, _ = line.strip('\n').split()        
                if word != '<#s>':
                    vocab_list.add(word)
    cPickle.dump(dict(enumerate(vocab_list)), open('vocab', "wb"))
    print "vocab generated successfully!"
