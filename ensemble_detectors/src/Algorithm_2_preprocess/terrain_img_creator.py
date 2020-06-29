#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:50:48 2019

@author: bisque
"""
import os
import cv2
import math
import numpy as np
import spectral as spy
import spectral.io.envi as envi
import shutil

import logging
import coloredlogs

# set the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aviris_data_loader")
coloredlogs.install(level='DEBUG', logger=logger)

DIRECTORY = '../../../single_detector/data'
FILES = [x for x in os.listdir(DIRECTORY)]
print(FILES)

def ignore_and_range(MAT, k=-9999.0):
    #ignore k , normalize and set range to (0,255)
    MAT[MAT==k]=0
    MAT = ((MAT-(MAT.min()))) / (MAT.max() - MAT.min())
    MAT *= 255
    
    return MAT

#%% Read each band and get the data accordingly.                            
def read_HSI_band(channel, hdr_img, extracted_HSI):
    for band in range(channel):
        print(band)
        extracted_HSI[:,:,band] = spy.open_image(hdr_img).read_band(band)
        
PROCESSEDDATA_PATH = "../../data/terrain_img_files"
for fname in FILES:
    sname = f'{fname}_img'  #spectral data
    hname = f'{sname}.hdr'  # header
    mname = f'{sname}_mask.png'  # manual annotated mask
    png_img = f'{DIRECTORY}/{fname}/{mname}'
    img_img = f'{DIRECTORY}/{fname}/{sname}'
    hdr_img = f'{DIRECTORY}/{fname}/{hname}'
    filename_dict = fname.split('_')[0]
    
    row_size, col_size, channel = spy.envi.open(hdr_img).shape
    extracted_HSI = np.zeros((row_size, col_size, channel))
    read_HSI_band(channel, hdr_img, extracted_HSI) 
    
    #reading each band and normalizing them.    
    r = ignore_and_range(extracted_HSI[:,:,0])
    g = ignore_and_range(extracted_HSI[:,:,1])
    b = ignore_and_range(extracted_HSI[:,:,2])

    # converting rgb image to grey scale to save to 1 channel and use in mrcnn    
    img_rgb = np.dstack((r ,g ,b))
    img_gray = cv2.cvtColor(np.uint8(img_rgb), cv2.COLOR_RGB2GRAY)
    
    # create directory and save the files
    tiles_folder = f'{PROCESSEDDATA_PATH}/{fname}'
    if not(os.path.isdir(tiles_folder)):
        os.mkdir(tiles_folder)
        print("\nDirectory", tiles_folder ," created.")
    elif os.path.isdir(tiles_folder):
        print("\nDirectory", tiles_folder ," already exists..deleting it")
        shutil.rmtree(tiles_folder)
        os.mkdir(tiles_folder)
        print("\nNew Directory", tiles_folder ," created.")
    
    img_filename = f'{tiles_folder}/{sname}.npy'
    np.save(img_filename, img_gray)
    
