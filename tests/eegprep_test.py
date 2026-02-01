#%%
import mne
from mne.preprocessing import ICA
from utils.eeg_mne2eeglab_epochs import eeg_mne2eeglab_epochs
from utils.iclabel import iclabel
from utils.pop_loadset import pop_loadset
from utils.ICL_feature_extractor import ICL_feature_extractor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
#%%
#Create Empty Dataframe 
columns = ["ID#"] + ["IC#"] + [f"PSD_{i+1}" for i in range(100)] + \
                    [f"AC_{i+1}" for i in range(100)] + \
                    [f"TOPO_{i+1}" for i in range(1024)]
df = pd.DataFrame(columns=columns)


# fpaths = ["C:\\Users\\goddardhudson\\Desktop\\H1004_allcond_ICA_TMPEEG.mat",
#           "C:\\Users\\goddardhudson\\Desktop\\H1007_allcond_ICA_TMPEEG.mat",
#           "C:\\Users\\goddardhudson\\Desktop\\H1009_allcond_ICA_TMPEEG.mat"]
rel_path = os.path.abspath("..\\data\\icmobi_feats\\raw_mats")
fpaths = [os.path.join(rel_path,"H1004_allcond_ICA_TMPEEG.mat"),
          os.path.join(rel_path,"H1007_allcond_ICA_TMPEEG.mat"),
          os.path.join(rel_path,"H1009_allcond_ICA_TMPEEG.mat")]

#%%
for path in fpaths:
    #%%
    # print("loading subject from path: %s",char(path))
    EEG = pop_loadset(path)
    features = ICL_feature_extractor(EEG, True)
    
    iclabel(EEG)

    topo = np.array(features[0]) #topo: (32, 32, 1, #IC)
    ac = np.array(features[1]) #ac: (1, 100, 1, #IC)
    psd = np.array(features[2]) #psd: (1, 100, 1, #IC)
    
    ## throwin in some visual data checks: 
    ## Topo: plt.imshow(topo[0].reshape(32, 32), cmap='viridis', interpolation='nearest')
    ## AC: plt.plot(ac[0], label='Autocorrelation 1')
    ## PSD: plt.plot(psd[0], label='Autocorrelation 1')
    
    ic_count = psd.shape[-1]

    flattened_images = []
    for i in range(topo.shape[3]):
        flattened_image = topo[:, :, 0, i].flatten()
        flattened_images.append(flattened_image)
    topo_flat = np.array(flattened_images)

    ac_flat = ac.reshape(ic_count, 100) #topo_flat: (#IC, 1024)
    psd_flat = psd.reshape(ic_count, 100) #ac_flat: (#IC, 100)

    ## throwin in some visual data checks: 
    ## Topo: plt.imshow(topo_flat[0].reshape(32, 32), cmap='viridis', interpolation='nearest')
    ## AC: plt.plot(ac_flat[0], label='Autocorrelation 1')
    ## PSD: plt.plot(psd_flat[0], label='Autocorrelation 1')
    
    # (test the conversion back to original shape)
    
    ## throwin in some visual data checks: 
    ## Topo: plt.imshow(topo_flat[0].reshape(32, 32), cmap='viridis', interpolation='nearest')
    ## AC: plt.plot(ac_flat[0], label='Autocorrelation 1')
    ## PSD: plt.plot(psd_flat[0], label='Autocorrelation 1')



    ## Revert Shapes: 
    

    ic_count = topo_flat.shape[0]

    #topo_unflattened = topo_flat.reshape(ic_count, 32, 32, 1).transpose(1, 2, 3, 0)  # (32, 32, 1, ic_count)
    #ac_unflattened = ac_flat.reshape( 1, 100, 1, ic_count)
    #psd_unflattened = psd_flat.reshape( 1, 100, 1, ic_count)

    data_chunk = np.hstack([psd_flat, ac_flat, topo_flat])
    chunk_df = pd.DataFrame(data_chunk, columns=columns[2:])


    chunk_df.insert(0, "ID#", os.path.basename(path))
    chunk_df.insert(0, "IC#", np.arange(1, ic_count + 1))

    df = pd.concat([df, chunk_df], ignore_index=True)
    
#df.to_csv('eegprep_test_features.csv', index=False)   
print(features)
# %%
