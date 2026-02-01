
#%%
from time import (time, gmtime, strftime)
import os
from shutil import rmtree
from os.path import isdir, isfile, join, basename
import pickle as pkl
from collections import OrderedDict
from copy import copy
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import webbrowser as wb
from scipy import io
import pandas as pd
import sys

from icldata import ICLabelDataset



#%%
labels = 'all'
dat_dir = os.path.abspath("G:\\Github\\icmobi_extension\\python_src\\data")
icl = ICLabelDataset(label_type=labels,datapath=dat_dir)
icl_data = icl.load_semi_supervised()
train_u, train_l, test, val, names = icl.load_semi_supervised()
t1 = train_l[0]


#Get ID, AC, and PSD
t1_ids = t1['ids']
t1_ac = t1['autocorr']
t1_psd = t1['psd']


#get topo and then pad with zeros icl function to make 32x32
#then flatten it from (x, 32, 32) to (x, 1024)
#do we want to get rid of this padding for training?
t1_topo = icl.pad_topo(t1['topo']).reshape(8651, -1)


#pad_topo
combined = np.concatenate(
    [t1_ids, t1_psd, t1_ac, t1_topo, labels],
    axis=1
)


#get first set of labels
labels = train_l[1][0]

#Create Pandas Dataframe
columns = (
    ["ID#"] + 
    ["IC#"] + 
    [f"PSD_{i+1}" for i in range(100)] + 
    [f"AC_{i+1}" for i in range(100)] + 
    [f"TOPO_{i+1}" for i in range(1024)] + 
    [f"LABEL_{i+1}" for i in range(7)]
)
df = pd.DataFrame(combined, columns=columns)

#Save to csv if Needed
#df.to_csv('trainset_labeled.csv', index=False)  
#df.head(10).to_csv("mini_trainset_labeled.csv", index=False)



# %%
