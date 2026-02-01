#%%
from icmobi_ext.utils.icldata import ICLabelDataset  # available from https://github.com/lucapton/ICLabel-Dataset
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn import metrics
from os.path import isdir, join
import pandas as pd
from scipy.io import savemat
from matplotlib import pyplot as plt
import itertools
import numpy as np
import os

#%%
seed1 = 1979
seed2 = 1776
seed3 = 1492
n_folds = 10
labels = 'all'
# icl_archs = [WeightedConvMANN]
# ilc_methods = [x.name + ' w/ acor'*y for x, y in itertools.product(icl_archs, range(2))]
other_archs = ICLabelDataset().load_classifications(2, np.array([[1, 1]])).keys()
# cls_map = {x: y for x, y in zip(ilc_methods + other_archs, range(len(ilc_methods + other_archs)))}
# cls_imap = {y: x for x, y in cls_map.iteritems()}
# cls_imap = [cls_imap[x] for x in range(len(cls_map))]

#-- load dataset
rel_path = os.path.abspath("..\\data\\")
icl = ICLabelDataset(label_type=labels,datapath=rel_path,seed=seed1)
# icl_data = icl.load_semi_supervised()
tmp_dat = icl.load_data()
#%% EXTRACT W/O FOR-LOOPS
train_u, train_l, test, val, names = icl.load_semi_supervised()
t1 = train_l[0]
t1_ids = t1['ids']



#%%
ilc_methods = [x.name + ' w/ acor'*y for x, y in itertools.product(icl_archs, range(2))]

cls_map = {x: y for x, y in zip(ilc_methods + other_archs, range(len(ilc_methods + other_archs)))}
cls_imap = {y: x for x, y in cls_map.iteritems()}
cls_imap = [cls_imap[x] for x in range(len(cls_map))]


scores = pd.DataFrame(columns=cols)
raw = {x: [[]] * n_folds for x in cls_imap}
raw.update({'label': [[]] * n_folds})
# train and extract performance statistics
for labels in ('all',):

    # load data
    icl = ICLabelDataset(label_type=labels, seed=seed1)
    icl_data = icl.load_semi_supervised()
    icl_data_val_labels = np.concatenate((icl_data[1][1][0], icl_data[3][1][0]), axis=0)
    icl_data_val_ilrlabels = np.concatenate((icl_data[1][1][1], icl_data[3][1][1]), axis=0)
    icl_data_val_ilrlabelscov = np.concatenate((icl_data[1][2][1], icl_data[3][2][1]), axis=0)

    # process topo maps
    topo_data = list()
    for it in range(4):
        temp = 0.99 * icl_data[it][0]['topo'] / np.abs(icl_data[it][0]['topo']).max(1, keepdims=True)
        topo_data.append(icl.pad_topo(temp).astype(np.float32).reshape(-1, 32, 32, 1))

    # generate mask
    mask = np.setdiff1d(np.arange(1024), icl.topo_ind)

    # K-fold
    kfold = StratifiedKFold(n_splits=n_folds, random_state=seed2)
    ind_fold = 0

    for ind_train_l, ind_test in kfold.split(icl_data_val_labels, icl_data_val_labels.argmax(1)):

        # create validation set
        sss = StratifiedShuffleSplit(1, len(ind_test), random_state=seed3)
        sss_gen = sss.split(icl_data_val_labels[ind_train_l], icl_data_val_labels[ind_train_l].argmax(1))
        ind_train_l_tr, ind_train_l_val = sss_gen.next()

        for use_autocorr in (False, True):

            # rescale features
            if use_autocorr:
                input_data = [[topo_data[x],
                               0.99 * icl_data[x][0]['psd'],
                               0.99 * icl_data[x][0]['autocorr'],
                               ] for x in range(4)]
            else:
                input_data = [[topo_data[x],
                               0.99 * icl_data[x][0]['psd'],
                               ] for x in range(4)]

            # create data fold
            temp = [np.concatenate((x, y), axis=0) for x, y in zip(input_data[1], input_data[3])]
            input_data[1] = [x[ind_train_l] for x in temp]                  # labeled train
            input_data[2] = [x[ind_train_l[ind_train_l_tr]] for x in temp]  # labeled train fold
            input_data[3] = [x[ind_train_l[ind_train_l_val]] for x in temp] # labeled validation fold
            input_data.append([x[ind_test] for x in temp])                  # test data
            test_ids = np.concatenate((icl_data[1][0]['ids'], icl_data[3][0]['ids']), axis=0)[ind_test]

            # create label fold
            train_labels = icl_data_val_labels[ind_train_l]
            train_labels_tr = icl_data_val_labels[ind_train_l[ind_train_l_tr]]
            train_labels_val = icl_data_val_labels[ind_train_l[ind_train_l_val]]
            test_labels = icl_data_val_labels[ind_test]

            train_ilrlabels = icl_data_val_ilrlabels[ind_train_l]
            train_ilrlabels_tr = icl_data_val_ilrlabels[ind_train_l[ind_train_l_tr]]
            train_ilrlabels_val = icl_data_val_ilrlabels[ind_train_l[ind_train_l_val]]
            test_ilrlabels = icl_data_val_ilrlabels[ind_test]

            train_ilrlabelscov = icl_data_val_ilrlabelscov[ind_train_l]
            train_ilrlabelscov_tr = icl_data_val_ilrlabelscov[ind_train_l[ind_train_l_tr]]
            train_ilrlabelscov_val = icl_data_val_ilrlabelscov[ind_train_l[ind_train_l_val]]
            test_ilrlabelscov = icl_data_val_ilrlabelscov[ind_test]

            # augment dataset by negating and/or horizontally flipping topo maps
            for it in range(5):
                input_data[it][0] = np.concatenate((input_data[it][0],
                                                    -input_data[it][0],
                                                    np.flip(input_data[it][0], 2),
                                                    -np.flip(input_data[it][0], 2)))
                for it2 in range(1, len(input_data[it])):
                    input_data[it][it2] = np.tile(input_data[it][it2], (4, 1))
            try:
                train_labels = np.tile(train_labels, (4, 1))
                train_labels_tr = np.tile(train_labels_tr, (4, 1))
                train_labels_val = np.tile(train_labels_val, (4, 1))
                test_labels = np.tile(test_labels, (4, 1))
                # ilr labels
                train_ilrlabels = np.tile(train_ilrlabels, (4, 1))
                train_ilrlabels_tr = np.tile(train_ilrlabels_tr, (4, 1))
                train_ilrlabels_val = np.tile(train_ilrlabels_val, (4, 1))
                test_ilrlabels = np.tile(test_ilrlabels, (4, 1))
                # ilr labels cov
                train_ilrlabelscov = np.tile(train_ilrlabelscov, (4, 1, 1))
                train_ilrlabelscov_tr = np.tile(train_ilrlabelscov_tr, (4, 1, 1))
                train_ilrlabelscov_val = np.tile(train_ilrlabelscov_val, (4, 1, 1))
                test_ilrlabelscov = np.tile(test_ilrlabelscov, (4, 1, 1))
            except ValueError:
                train_labels = 4 * train_labels
                train_labels_tr = 4 * train_labels_tr
                train_labels_val = 4 * train_labels_val
                test_labels = 4 * test_labels
                # ilr labels
                train_ilrlabels = 4 * train_ilrlabels
                train_ilrlabels_tr = 4 * train_ilrlabels_tr
                train_ilrlabels_val = 4 * train_ilrlabels_val
                test_ilrlabels = 4 * test_ilrlabels
                # ilr labels cov
                train_ilrlabelscov = 4 * train_ilrlabelscov
                train_ilrlabelscov_tr = 4 * train_ilrlabelscov_tr
                train_ilrlabelscov_val = 4 * train_ilrlabelscov_val
                test_ilrlabelscov = 4 * test_ilrlabelscov

            test_ids = np.tile(test_ids, (4, 1))

            # describe features and name
            additional_features = OrderedDict([('psd_med', input_data[1][1].shape[1])])
            name = 'ICLabel2_' + labels
            if use_autocorr:
                additional_features['autocorr'] = input_data[1][2].shape[1]
                name += '_autocorr'

            name += '_cv' + str(ind_fold)

            raw['label'][ind_fold] = test_labels

# %%
