#%% IMPORTS
#09/15/2024 - HG Import psutil for memory debugging
# import os
# import sys
import numpy as np
import scipy
from icmobi_ext.utils.icldata import ICLabelDataset
from sklearn.model_selection import KFold
# from icmobi_ext.eegprep_icmobi.iclabel_net import ICLabelNet
from icmobi_ext.icmobi_model import ICLabelNet
import torch
import os
# from icmobi_model import icldata, icmobi_model, plot_functions  # code adapted from https://github.com/lucapton/ICLabel-Dataset
#%% TRAINING FUNCTIONS

#%% MAIN K-FOLD PROCESS
if __name__ == "__main__":
    #%% DOWNLOAD ICLABEL TRAIN FEATURE SETS
    # Setup dataset & Initialize models to be constructed 
    rel_path = os.path.abspath("..\\data\\")
    
    icl = ICLabelDataset(datapath=rel_path)
    icl.download_trainset_features()
    download_feats = ['train_labels','train_features','test_labels','test_features','database','classifications']
    icl.check_for_download(download_feats)
    seed1 = 1979
    seed2 = 1776
    seed3 = 1492
    n_folds = 10
    #%%
    # Instantiate the model with example input shapes
    # model = ICLabelNet()
    # model.build([(None, 32, 32, 1), (None, 32, 32, 1), (None, 32, 32, 1)])
    # model.summary()
    # Assume you have your dataset
    # images, psds, autocorr are numpy arrays of your data
    # labels is the ground truth label array
    # For this example, assume they are loaded with shapes suitable for the model

    # load data
    icl = ICLabelDataset(label_type='all', seed=seed1)

    # look into debug of line 903 in icldata
    icl_data = icl.load_semi_supervised()
    #%%
    images =  icl_data[1][0]['topo']
    psds = icl_data[1][0]['psd']
    autocorr = icl_data[1][0]['autocorr']
    # labels = icl_data[1][0]['labels']
    labels = np.concatenate((icl_data[1][1][0], icl_data[3][1][0]), axis=0)

    icl_data_val_labels = np.concatenate((icl_data[1][1][0], icl_data[3][1][0]), axis=0)
    icl_data_val_ilrlabels = np.concatenate((icl_data[1][1][1], icl_data[3][1][1]), axis=0)
    icl_data_val_ilrlabelscov = np.concatenate((icl_data[1][2][1], icl_data[3][2][1]), axis=0)
    
    # model = ICLabelNet('netICL.mat')
    # image_mat = scipy.io.loadmat('net_vars.mat')['in_image']
    # psdmed_mat = scipy.io.loadmat('net_vars.mat')['in_psdmed']
    # autocorr_mat = scipy.io.loadmat('net_vars.mat')['in_autocorr']
    # # assuming third dimension is trivial and last dimension is channel. First two dimensions (32 x 32) are size of topoplot
    # image = torch.tensor(image_mat).permute(-1, 2, 0, 1)
    # print('image shape', image.shape)
    # psdmed = torch.tensor(psdmed_mat).permute(-1, 2, 0, 1)
    # print('psd shape', psdmed.shape)
    # autocorr = torch.tensor(autocorr_mat).permute(-1, 2, 0, 1)
    # print('autocorr shape', autocorr.shape)
    # output = model(image, psdmed, autocorr)
    # print(output.shape)

    # # save the output to a mat file
    # scipy.io.savemat('output4.mat', {'output': output.detach().numpy()})
    

# %%
