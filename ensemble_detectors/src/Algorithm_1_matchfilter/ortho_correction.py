#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:15:38 2019

@author: Satish
"""
# doing the ortho-correction on the processed data from matchedFilter

import os
import numpy as np
import spectral as spy
import spectral.io.envi as envi
import spectral.algorithms as algo
from spectral.algorithms.detectors import MatchedFilter, matched_filter

import logging
import coloredlogs
import json
import shutil
import statistics

# set the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aviris_data_loader")
coloredlogs.install(level='DEBUG', logger=logger)

DIRECTORY = "../../data/raw_data"

#manual offset file load
try:
    #Read the manually computed offset file
    f = open('./manual_offset.json')
    offset_data = json.load(f)
    OFFSET_DICT = offset_data['OFFSET_DICT']
except:
    print("No manual offset file found")
    pass

FILES = []
for x in os.listdir(DIRECTORY):        
    if(os.path.isdir(os.path.join(DIRECTORY, x))):
        FILES.append(x)
print(FILES)

#%% return image object
def image_obj(hdr, img):
    "create a object of the image corresponding to certain header"
    
    head = envi.read_envi_header(hdr)
    param = envi.gen_params(head)
    param.filename = img   # spectral data file corresponding to .hdr file
    
    interleave = head['interleave']
    if (interleave == 'bip' or interleave == 'BIP'):
        print("it is a bip")
        from spectral.io.bipfile import BipFile
        img_obj = BipFile(param, head)
        
    if (interleave == 'bil' or interleave == 'BIL'):
        print("It is a bil file")
        from spectral.io.bilfile import BilFile
        img_obj = BilFile(param, head)
        
    return img_obj

# Use this fucntion in case you have data other than the custom dataset
def ideal_ortho_correction(glt: np.ndarray, img: np.ndarray, b_val=0.0, output=None) -> np.ndarray:
    """does the ortho-correction of the file
    glt: 2L, world-relative coordinates L1: y (rows), L2: x (columns)
    img: 1L, unrectified, output from matched filter
    output: 1L, rectified version of img, with shape: glt.shape
    """
    if output is None:
        output = np.zeros((glt.shape[0], glt.shape[1]))
    if not np.array_equal(output.shape, [glt.shape[0], glt.shape[1]]):
        print("image dimension of output arrary do not match the GLT file")
    # getting the absolute even if GLT has negative values
    # magnitude
    glt_mag = np.absolute(glt) 
    # GLT value of zero means no data, extract this because python has zero-indexing.
    glt_mask = np.all(glt_mag==0, axis=2)
    output[glt_mask] = b_val
    glt_mag[glt_mag>(img.shape[0]-1)] = 0
    # now check the lookup and fill in the location, -1 to map to zero-indexing
    # output[~glt_mask] = img[glt_mag[~glt_mask, 1] - 1, glt_mag[~glt_mask, 0] - 1]
    output[~glt_mask] = img[glt_mag[~glt_mask, 1]-1, glt_mag[~glt_mask, 0]-1]
    
    return output

def custom_ortho_correct_for_data(file_name, glt: np.ndarray, img: np.ndarray, OFFSET_DICT, b_val=0.0, output=None) -> np.ndarray:
    """does the ortho-correction of the file
    glt: 2L, world-relative coordinates L1: y (rows), L2: x (columns)
    img: 1L, unrectified, output from matched filter
    output: 1L, rectified version of img, with shape: glt.shape
    """
    if output is None:
        output = np.zeros((glt.shape[0], glt.shape[1]))
    if not np.array_equal(output.shape, [glt.shape[0], glt.shape[1]]):
        print("image dimension of output arrary do not match the GLT file")
    
    print(file_name)
    if file_name in OFFSET_DICT.keys():
        offset_mul = OFFSET_DICT[file_name]
    else:
        return 0
    print(offset_mul)
    off_v = int(offset_mul*1005)
    img_readB = img[off_v:img.shape[0],:]
    img_readA = img[0:off_v,:]
    img_read = np.vstack((img_readB,img_readA))
    if ((glt.shape[0]-img.shape[0])>0):
        print("size mismatch. Fixing it...")
        completion_shape = np.zeros((glt.shape[0]-img.shape[0], img.shape[1]))
        img_read = np.vstack((img_read,completion_shape))
    print(img_read.shape)
    # getting the absolute even if GLT has negative values
    # magnitude
    glt_mag = np.absolute(glt)
    # GLT value of zero means no data, extract this because python has zero-indexing.
    glt_mask = np.all(glt_mag==0, axis=2)
    output[glt_mask] = b_val
    glt_mag[glt_mag>(img.shape[0]-1)] = 0
    # now check the lookup and fill in the location, -1 to map to zero-indexing
    output[~glt_mask] = img_read[glt_mag[~glt_mask,1]-1, glt_mag[~glt_mask,0]-1]
    
    return output

#%% load file and rectify it in each band
for fname in FILES:
    
    fname_glt = fname.split("_")[0]
    sname_glt = f'{fname_glt}_rdn_glt'  #geo-ref file for ortho-correction
    hname_glt = f'{sname_glt}.hdr'      #header file
    glt_img = f'{DIRECTORY}/{fname}/{sname_glt}'
    glt_hdr = f'{DIRECTORY}/{fname}/{hname_glt}'
    print(glt_img, glt_hdr)
    mf_folder = f'{DIRECTORY}/{fname}/{fname_glt}_rdn_v1f_clip_mfout'    

    try:
        if (fname_glt not in OFFSET_DICT.keys()):
            continue
        if (os.path.exists(glt_hdr)):
            glt_data_obj = image_obj(glt_hdr, glt_img)
            glt = glt_data_obj.read_bands([0,1])
        else:
            continue
    except:
        pass

    #mf_rect_path = f'/media/data/satish/detector_bank_input/corrected_output'
    mf_rect_folder = f'{DIRECTORY}/{fname}/{fname_glt}_rect'
    if not(os.path.isdir(mf_rect_folder)):
        os.mkdir(mf_rect_folder)
        print("\nDirectory", mf_rect_folder ," created.")
    elif os.path.isdir(mf_rect_folder):
        print("\nDirectory", mf_rect_folder ," already exists..deleting it")
        shutil.rmtree(mf_rect_folder)
        os.mkdir(mf_rect_folder)
        print("\nNew Directory", mf_rect_folder ," created.")
    
    for mfname in os.listdir(mf_folder):
        print("Ortho-correcting file", mfname)
        mf_filename = f'{mf_folder}/{mfname}'
        img_unrect = np.load(mf_filename)
        print(img_unrect.shape)
        '''
        use this function in case you have any other dataset, the custom_ortho_correct_for_data
        function uses the OFFSET_DICT to correct the row positions in each band.
        rect_img = ideal_ortho_correction(fname_glt, glt, img_unrect)
        '''
        rect_img = custom_ortho_correct_for_data(fname_glt, glt, img_unrect, OFFSET_DICT)
        rect_filename = f'{mf_rect_folder}/{mfname}'
        np.save(rect_filename, rect_img)
            
