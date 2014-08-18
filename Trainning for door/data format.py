import re
import pickle
import cPickle
import os
import numpy as n
from math import sqrt
filenames='.\data\data_batch_1'
out_file=file(filenames,'wb')
in_file=open('data_batch_01','r+')
dd=cPickle.load(in_file)
dd['batch_label']='training batch 1 of 1'
print dd['batch_label']
pickle.dump(dd,out_file)
in_file.close()
print 'finish'
