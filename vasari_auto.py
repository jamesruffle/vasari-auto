# This is the codebase for VASARI-auto
###	vasari-auto.py | a pipeline for automated VASARI characterisation of glioma.
###	Copyright 2024 James Ruffle, High-Dimensional Neurology, UCL Queen Square Institute of Neurology.
###	This program is licensed under the APACHE 2.0 license.
###	This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
### This program is not intended for clinical use of any kind.
###	See the License for more details.
###	This code is part of the repository https://github.com/james-ruffle/vasari-auto
###	Correspondence to Dr James K Ruffle by email: j.ruffle@ucl.ac.uk

#Import packages
import glob
import numpy as np
import os
import pandas as pd
import shutil
import errno
import subprocess
from datetime import datetime
from tqdm import tqdm
import argparse
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import seaborn as sns
from sklearn.metrics import *
import time
from skimage.morphology import skeletonize
import matplotlib.ticker as mticker
from scipy import stats
pd.set_option('display.max_rows', 500)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def get_vasari_features(file,atlases='/atlas_masks/',verbose=False,enhancing_label=3,nonenhancing_label=1,oedema_label=2,z_dim=-1,cf=1,t_ependymal=5000,t_wm=100,resolution=1,midline_thresh=5,enh_quality_thresh=15,cyst_thresh=50,cortical_thresh=1000,focus_thresh=30000,num_components_bin_thresh=10,num_components_cet_thresh=15):
    """
    #Required argument
    file - NIFTI segmentation file with binary lesion labels
    atlases - atlas path for location derivation
    
    #Optional hyperparmeters
    verbose - whether to enable verbose logging, default=False
    enhancing_label - the integer value of enhancing tumour within file, default=3
    nonenhancing_label - the integer value of nonenhancing tumour within file, default=1
    oeedema_label - the integer value of nonenhancing tumour within file, default=2
    z_dim - the dimension of the Z axis within file, default=-1, which assumes MNI template registration
    cf - correction factor for ambiguity in voxel quantification, default=1
    t_ependymal - threshold for lesion involvement within the ependyma, this can be customised depending on the voxel resolution you are operating in, default=5000
    t_wm - threshold for lesion involvement within the wm, this can be customised depending on the voxel resolution you are operating in, default=100
    resolution - volumetric voxel resolution, this is important for derivation of F11 - thickness of enhancing margin, default=1 (1mm x 1mm x 1mm resolution)
    midline_thresh - threshold for number of diseased voxels that can cross the midline to be quantified as a lesion definitively crossing the midline, default=5
    enh_quality_thresh - threshold for determining the quality of lesion enhancement by volume approximation. Please note ideally this feature would utilise source imaging but in its prototype format uses anonymised segmentation data only, default=15
    cyst_thresh - threshold for determining the presence of cysts based on a heuristic of nCET detection, default=50
    cortical_thresh - threshold for determining cortex involvement, default=1000
    focus_thresh - threshold for determining a principle side of involvement, this will vary depending on resolution, default=30000
    num_components_bin_thresh - threshold for quantifying a multifocal lesion, default = 10
    num_components_cet_thresh - threshold for satellite lesions, default=15
    """
    
    start_time = time.time()
    
    if verbose:
        print('Please note that this software is in beta and utilises only irrevocably anonymised lesion masks.\nVASARI features that require source data shall not be derived and return NaN in this software version')
        print('')
        print('Working on: '+str(file))
        print('')

    ##derive anatomy masks - this is for automated location (F1)
    brainstem = atlases+'brainstem.nii.gz'
    brainstem = nib.load(brainstem)
    brainstem_array = np.asanyarray(brainstem.dataobj)
    brainstem_vol = np.sum(brainstem_array)
    
    frontal_lobe = atlases+'frontal_lobe.nii.gz'
    frontal_lobe = nib.load(frontal_lobe)
    frontal_lobe_array = np.asanyarray(frontal_lobe.dataobj)
    frontal_lobe_vol = np.sum(frontal_lobe_array)
    
    insula = atlases+'insula.nii.gz'
    insula = nib.load(insula)
    insula_array = np.asanyarray(insula.dataobj)
    insula_vol = np.sum(insula_array)
    
    occipital = atlases+'occipital.nii.gz'
    occipital = nib.load(occipital)
    occipital_array = np.asanyarray(occipital.dataobj)
    occipital_vol = np.sum(occipital_array)
    
    parietal = atlases+'parietal.nii.gz'
    parietal = nib.load(parietal)
    parietal_array = np.asanyarray(parietal.dataobj)
    parietal_vol = np.sum(parietal_array)
    
    temporal = atlases+'temporal.nii.gz'
    temporal = nib.load(temporal)
    temporal_array = np.asanyarray(temporal.dataobj)
    temporal_vol = np.sum(temporal_array)
    
    thalamus = atlases+'thalamus.nii.gz'
    thalamus = nib.load(thalamus)
    thalamus_array = np.asanyarray(thalamus.dataobj)
    thalamus_vol = np.sum(thalamus_array)
    
    corpus_callosum = atlases+'corpus_callosum.nii.gz'
    corpus_callosum = nib.load(corpus_callosum)
    corpus_callosum_array = np.asanyarray(corpus_callosum.dataobj)
    corpus_callosum_vol = np.sum(corpus_callosum_array)
    
    ventricles = atlases+'ventricles.nii.gz'
    ventricles = nib.load(ventricles)
    ventricles_array = np.asanyarray(ventricles.dataobj)
    ventricles_vol = np.sum(ventricles_array)
    
    internal_capsule = atlases+'internal_capsule.nii.gz'
    internal_capsule = nib.load(internal_capsule)
    internal_capsule_array = np.asanyarray(internal_capsule.dataobj)
    internal_capsule_vol = np.sum(internal_capsule_array)
    
    cortex = atlases+'cortex.nii.gz'
    cortex = nib.load(cortex)
    cortex_array = np.asanyarray(cortex.dataobj)
    cortex_vol = np.sum(cortex_array)
    
    segmentation = nib.load(file)
    segmentation_array = np.asanyarray(segmentation.dataobj)
    
    if verbose:
        print('Running voxel quantification per tissue class')
    total_lesion_burden = np.count_nonzero(segmentation_array)
    enhancing_voxels = np.count_nonzero(segmentation_array == enhancing_label)
    nonenhancing_voxels = np.count_nonzero(segmentation_array == nonenhancing_label)
    oedema_voxels = np.count_nonzero(segmentation_array == oedema_label)
    
    if verbose:
        print('Deriving number of components')
    labeled_array, num_components = label(segmentation_array)
    
    if verbose:
        print('Determining laterality')
        #print('Note - if experiencing unexpected axis flipping for lesion laterality, check lesion registration space. This code assumes MNI template registration')
    temp = segmentation_array.nonzero()[0]
    right_hemisphere=len(temp[temp<int(segmentation_array.shape[z_dim]/2)])
    left_hemisphere=len(temp[temp>int(segmentation_array.shape[z_dim]/2)])
    if right_hemisphere>left_hemisphere:
        side='Right'
    if right_hemisphere<left_hemisphere:
        side='Left'
    if right_hemisphere>focus_thresh and left_hemisphere>focus_thresh:
        side='Bilateral'
    # if verbose:
        # print(right_hemisphere)
        # print(left_hemisphere)

    if verbose:
        print('Determining proportions')
    segmentation_array[segmentation_array==oedema_label]=0
    segmentation_array[segmentation_array==enhancing_label]=1
    segmentation_array[segmentation_array==nonenhancing_label]=1
    
    if segmentation_array.sum()==0:
        if verbose:
            print('No lesion detected, falling back to oedema label for closer inspection')
        segmentation_array = np.asanyarray(segmentation.dataobj)
        segmentation_array[segmentation_array!=oedema_label]=0

    prop_in_brainstem = len((segmentation_array*brainstem_array).nonzero()[0])/(segmentation_array.sum()+cf)
    prop_in_frontal_lobe = len((segmentation_array*frontal_lobe_array).nonzero()[0])/(segmentation_array.sum()+cf)
    prop_in_insula = len((segmentation_array*insula_array).nonzero()[0])/(segmentation_array.sum()+cf)
    prop_in_occipital = len((segmentation_array*occipital_array).nonzero()[0])/(segmentation_array.sum()+cf)
    prop_in_parietal = len((segmentation_array*parietal_array).nonzero()[0])/(segmentation_array.sum()+cf)
    prop_in_temporal = len((segmentation_array*temporal_array).nonzero()[0])/(segmentation_array.sum()+cf)
    prop_in_thalamus = len((segmentation_array*thalamus_array).nonzero()[0])/(segmentation_array.sum()+cf)
    prop_in_cc = len((segmentation_array*corpus_callosum_array).nonzero()[0])/(segmentation_array.sum()+cf)

    d = {'ROI': ['Brainstem','Frontal Lobe','Insula','Occipital Lobe','Parietal Lobe','Temporal Lobe','Thalamus','Corpus callosum'], 
         'prop': [prop_in_brainstem,prop_in_frontal_lobe,prop_in_insula,prop_in_occipital,prop_in_parietal,prop_in_temporal,prop_in_thalamus,prop_in_cc]}
    
    vols = pd.DataFrame(data=d)
    vols = vols.sort_values(by='prop',ascending=False).reset_index(drop=True)
    
    if verbose:
        print(vols)
    
    proportion_enhancing = (enhancing_voxels/total_lesion_burden)*100
    proportion_nonenhancing = (nonenhancing_voxels/total_lesion_burden)*100
    proportion_oedema = (oedema_voxels/total_lesion_burden+.1)*100
    
    enhancement_quality = 1
    if proportion_enhancing>0: #heuristic of if model segments more than 10% voxels are enhancing
        if proportion_enhancing>enh_quality_thresh:
            enhancement_quality=3
        else:
            enhancement_quality=2
    
    if verbose:
        print('Determining ependymal involvement')
    if len((segmentation_array*ventricles_array).nonzero()[0])>=t_ependymal:
        ependymal=2

    if len((segmentation_array*ventricles_array).nonzero()[0])<t_ependymal:
        ependymal=1

    if verbose:
        print('Determining white matter involvemenet')
    deep_wm='None'
    if len((segmentation_array*brainstem_array).nonzero()[0])>=t_wm:
        deep_wm='Brainstem'

    if len((segmentation_array*corpus_callosum_array).nonzero()[0])>=t_wm:
        deep_wm='Corpus Callosum'

    if len((segmentation_array*internal_capsule_array).nonzero()[0])>=t_wm:
        deep_wm='Internal Capsule'
    deep_wm_f = np.nan
    if deep_wm=='None':
        deep_wm_f=1
    if deep_wm!='None':
        deep_wm_f=2
        
    if verbose:
        print('Determining cortical involvement')

    cortical_lesioned_voxels = len((segmentation_array*cortex_array).nonzero()[0])
    cortical_lesioned_voxels_f = np.nan
    if cortical_lesioned_voxels>cortical_thresh:
        cortical_lesioned_voxels_f=2
    if cortical_lesioned_voxels<=cortical_thresh:
        cortical_lesioned_voxels_f=1
    if verbose:
        print('Cortically lesioned voxels '+str(cortical_lesioned_voxels))

    if verbose:
        print('Determining midline involvement')
    nCET_cross_midline=False
    nCET = np.asanyarray(segmentation.dataobj)
    nCET[nCET!=nonenhancing_label]=0
    nCET[nCET>0]=1
    temp = nCET.nonzero()[0]
    right_hemisphere=len(temp[temp<int(segmentation_array.shape[z_dim]/2)])
    left_hemisphere=len(temp[temp>int(segmentation_array.shape[z_dim]/2)])
    if right_hemisphere>midline_thresh and left_hemisphere>midline_thresh:
        nCET_cross_midline=True
    nCET_cross_midline_f = np.nan
    if nCET_cross_midline==True:
        nCET_cross_midline_f=3
    if nCET_cross_midline==False:
        nCET_cross_midline_f=2
    
    CET_cross_midline=False
    CET = np.asanyarray(segmentation.dataobj)
    CET[CET!=enhancing_label]=0
    CET[CET>0]=1
    temp = CET.nonzero()[0]
    right_hemisphere=len(temp[temp<int(segmentation_array.shape[z_dim]/2)])
    left_hemisphere=len(temp[temp>int(segmentation_array.shape[z_dim]/2)])
    if right_hemisphere>midline_thresh and left_hemisphere>midline_thresh:
        CET_cross_midline=True
    CET_cross_midline_f = np.nan
    if CET_cross_midline==True:
        CET_cross_midline_f=3
    if CET_cross_midline==False:
        CET_cross_midline_f=2

    if verbose:
        print('Deriving enhancing satellites')
    labeled_array, num_components_cet = label(CET)
    num_components_cet_f = np.nan
    if num_components_cet>num_components_cet_thresh:
        num_components_cet_f =2
    else:
        num_components_cet_f=1
    
    if verbose:
        print('Deriving cysts')
    labeled_array, num_components_ncet = label(nCET)
    num_components_ncet_f =1
    if num_components_ncet>cyst_thresh:
        num_components_ncet_f=2
        
    if verbose:
        print('Cyst count '+str(num_components_ncet))
        
    if verbose:
        print('Deriving enhancement thickness')
        
    
    enhancing_skeleton = skeletonize(CET)
    allpixels = np.count_nonzero(CET)
    skeletonpixels = np.count_nonzero(enhancing_skeleton)
    
    if allpixels>0:
        enhancing_thickness = allpixels/(skeletonpixels+1e-9)
    if allpixels==0:
        enhancing_thickness=0
    enhancing_thickness_f = np.nan
    ll=3
    if enhancing_thickness<ll:
        enhancing_thickness_f=3
    if enhancing_thickness>=ll:
        enhancing_thickness_f=4
    if enhancing_thickness>=ll and nonenhancing_voxels==0:
        enhancing_thickness_f=5
        
    if verbose:
        print('Enhancing thickness: '+str(enhancing_thickness))
    
    if verbose:
        print('Converting raw values to VASARI dictionary features')
    F1_dict = {'Frontal Lobe':1,'Temporal Lobe':2,'Insula':3,'Parietal Lobe':4,'Occipital Lobe':5,'Brainstem':6,'Corpus callosum':7,'Thalamus':8}
    F2_dict = {'Right':1,'Left':3,'Bilateral':2}

    proportion_enhancing_f = np.nan
    if proportion_enhancing<=5:
        proportion_enhancing_f = 3
    if 5 < proportion_enhancing <= 33:
        proportion_enhancing_f = 4
    if 33 < proportion_enhancing <= 67:
        proportion_enhancing_f = 5
    if 67 < proportion_enhancing <= 100:
        proportion_enhancing_f = 6
        
    proportion_nonenhancing_f = np.nan
    if proportion_nonenhancing<=5:
        proportion_nonenhancing_f = 3
    if 5 < proportion_nonenhancing <= 33:
        proportion_nonenhancing_f = 4
    if 33 < proportion_nonenhancing <= 67:
        proportion_nonenhancing_f = 5
    if 67 < proportion_nonenhancing <= 95:
        proportion_nonenhancing_f = 6  
    if 95 < proportion_nonenhancing <= 99.5:
        proportion_nonenhancing_f = 7
    if proportion_nonenhancing >99.5: #allow for small segmentation variation
        proportion_nonenhancing_f = 8 
        
    proportion_necrosis_f = np.nan
    if proportion_nonenhancing==0:
        proportion_necrosis_f=2
    if 0<proportion_nonenhancing<=5:
        proportion_necrosis_f=3
    if 5<proportion_nonenhancing<=33:
        proportion_necrosis_f=4
    if 33<proportion_nonenhancing<=67:
        proportion_necrosis_f=5
        
    segmentation_array_binary = segmentation_array.copy()
    segmentation_array_binary[segmentation_array_binary>0]=1
    labeled_array_bin, num_components_bin = label(segmentation_array_binary)
    f9_multifocal = 1
    if num_components_bin>num_components_bin_thresh:
        f9_multifocal=2
    if verbose:
        print('Number of lesion components: '+str(num_components_bin))
        
    proportion_oedema_f=np.nan
    if verbose:
        print('prop oedema '+str(proportion_oedema))
    
    if proportion_oedema==0:
        proportion_oedema_f=2
    if 0<proportion_oedema<=5:
        proportion_oedema_f=3
    if 5<proportion_oedema<=33:
        proportion_oedema_f=4
    if 33<proportion_oedema:
        proportion_oedema_f=5
    
        
    end_time = time.time()
    time_taken = (end_time - start_time)
    if verbose:
        print("Time taken: "+ str(time_taken)+" seconds")
        
    if verbose:
        print('')
        print('Complete! Generating output...')
        
    col_names = ['filename', 'reporter', 'time_taken_seconds',
           'F1 Tumour Location', 'F2 Side of Tumour Epicenter',
           'F3 Eloquent Brain', 'F4 Enhancement Quality',
           'F5 Proportion Enhancing', 'F6 Proportion nCET',
           'F7 Proportion Necrosis', 'F8 Cyst(s)', 'F9 Multifocal or Multicentric',
           'F10 T1/FLAIR Ratio', 'F11 Thickness of enhancing margin',
           'F12 Definition of the Enhancing margin',
           'F13 Definition of the non-enhancing tumour margin',
           'F14 Proportion of Oedema', 'F16 haemorrhage', 'F17 Diffusion',
           'F18 Pial invasion', 'F19 Ependymal Invasion',
           'F20 Cortical involvement', 'F21 Deep WM invasion', 
                 'F22 nCET Crosses Midline', 'F23 CET Crosses midline',
                 'F24 satellites',
           'F25 Calvarial modelling', 'COMMENTS']
        

    result = pd.DataFrame(columns=col_names)
    result.loc[len(result)] = {'filename':file,
                               'reporter':'VASARI-auto',
                              'time_taken_seconds':time_taken,
                              'F1 Tumour Location':F1_dict[vols.iloc[0,0]], #vols.iloc[0,0],
                              'F2 Side of Tumour Epicenter':F2_dict[side],
                              'F3 Eloquent Brain':np.nan, #unsupported in current version
                              'F4 Enhancement Quality':enhancement_quality,
                              'F5 Proportion Enhancing':proportion_enhancing_f,
                              'F6 Proportion nCET':proportion_nonenhancing_f,
                              'F7 Proportion Necrosis':proportion_necrosis_f,
                              'F8 Cyst(s)':np.nan, #unsupported in current version num_components_ncet_f,
                                'F9 Multifocal or Multicentric':f9_multifocal,
                               'F10 T1/FLAIR Ratio':np.nan,  #unsupported in current version
                               'F11 Thickness of enhancing margin':enhancing_thickness_f,
                               'F12 Definition of the Enhancing margin':np.nan,  #unsupported in current version
                               'F13 Definition of the non-enhancing tumour margin':np.nan,  #unsupported in current version
                               'F14 Proportion of Oedema': proportion_oedema_f,
                               'F16 haemorrhage':np.nan,  #unsupported in current version
                               'F17 Diffusion':np.nan,  #unsupported in current version
                               'F18 Pial invasion':np.nan, #unsupported in current version
                               'F19 Ependymal Invasion':ependymal, 
                               'F20 Cortical involvement':cortical_lesioned_voxels_f,
                               'F21 Deep WM invasion':deep_wm_f, 
                               'F22 nCET Crosses Midline':nCET_cross_midline_f,
                               'F23 CET Crosses midline':CET_cross_midline_f,
                               'F24 satellites':num_components_cet_f, 
                               'F25 Calvarial modelling':np.nan, #unsupported in current version
                               'COMMENTS':'Please note that this software is in beta and utilises only irrevocably anonymised lesion masks. VASARI features that require source data shall not be derived and return NaN'
                              }
    return result