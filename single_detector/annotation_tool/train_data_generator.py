#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#file usage : python train_data_generator.py ../data
"""
Created on Wed Apr  3 14:50:48 2019

@author: satish
"""

import sys
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

DIRECTORY = "{}/".format(sys.argv[1])
FILES = [x for x in os.listdir(DIRECTORY)]
print(FILES)

#global max and minimum values accross the dataset, calculated manually first
GLOBAL_RAD = {
    "min": -140404.0625,
    "max": 146026.140625
    }

def ignore_and_range(MAT, k=-9999.0):
    """
    ignore k , normalize and set range to (0,255)
    """
    MAT[MAT==k]=0
    MAT = ((MAT-(MAT.min()))) / (MAT.max() - MAT.min())
    MAT *= 255
    
    return MAT

#%% Read each band data                             
def read_HSI_band(channel, hdr_img, extracted_HSI):
    for band in range(channel):
        print("Reading band :", band)
        extracted_HSI[:,:,band] = spy.open_image(hdr_img).read_band(band)

#%% Generating the RGB image and appending with local and global norm bands
PROCESSEDDATA_PATH = "../src/custom-mask-rcnn-detector/ch4_data/train_data"
if not (os.path.isdir(PROCESSEDDATA_PATH)):
	os.mkdir(PROCESSEDDATA_PATH)
	print("Directory", PROCESSEDDATA_PATH, "created")
elif os.path.isdir(PROCESSEDDATA_PATH):
    print("\nDirectory", PROCESSEDDATA_PATH ," already exists, adding files to it")

for fname in FILES:
    sname = f'{fname}_img'  #spectral data
    hname = f'{sname}.hdr'  # header
    mname = f'{sname}_mask.png'  # manual annotated mask
    png_img = f'{DIRECTORY}/{fname}/{mname}'
    img_img = f'{DIRECTORY}/{fname}/{sname}'
    hdr_img = f'{DIRECTORY}/{fname}/{hname}'
    row_size, col_size, channel = spy.envi.open(hdr_img).shape
    extracted_HSI = np.zeros((row_size, col_size, channel))
    read_HSI_band(channel, hdr_img, extracted_HSI) 
    
    #reading each band and normalizing them. 
    r = ignore_and_range(extracted_HSI[:,:,0])
    g = ignore_and_range(extracted_HSI[:,:,1])
    b = ignore_and_range(extracted_HSI[:,:,2])
    radiance = extracted_HSI[:,:,3].copy() 
    
    radiance[radiance==-9999.0]=0 # removing the reduntant value
    
    #global normalization of the 4th band that will be fed to mrcnn
    rad_diff = GLOBAL_RAD['max'] - GLOBAL_RAD['min']
    radiance_global = (radiance-GLOBAL_RAD['min']) / rad_diff
    
    #local normalization of the 4th band
    l_diff = radiance.max()-radiance.min()
    radiance_local = (radiance-radiance.min())/l_diff

    # converting rgb image to grey scale to save to 1 channel and use in mrcnn    
    img_rgb = np.dstack((r ,g ,b))
    img_gray = cv2.cvtColor(np.uint8(img_rgb), cv2.COLOR_RGB2GRAY)
    
    # 3 channel input for mrcnn 1.gray,2.ch4local_norm,3.ch4global_norm
    img_mrcnn = np.dstack((img_gray, radiance_local, radiance_global))

    # creating tiles folder and saving the whole file in tiles
    tiles_folder = f'{PROCESSEDDATA_PATH}/{fname}_img_radiance_tiles'
    if not(os.path.isdir(tiles_folder)):
        os.mkdir(tiles_folder)
        print("\nDirectory", tiles_folder ," created.")
    elif os.path.isdir(tiles_folder):
        print("\nDirectory", tiles_folder ," already exists..deleting it")
        shutil.rmtree(tiles_folder)
        os.mkdir(tiles_folder)
        print("\nNew Directory", tiles_folder ," created.")

    img_shape = img_mrcnn.shape
    print("shape of image to be tiled", img_shape)
    tile_size = (1024, 1024)
    offset = (512, 512)
    img_name = f'{sname}_img_radiance'
    
        
    for i in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
        for j in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
            cropped_tile = img_mrcnn[offset[1]*i:min(offset[1]*i+tile_size[1], \
                              img_shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], \
                                       img_shape[1])]

            check_tile = f'{img_name}_{i}_{j}.npy'
            
            if(check_tile not in tiles_folder):
                np.save(os.path.join(tiles_folder, (img_name + "_" + str(i) \
                + "_" + str(j))), cropped_tile)
   
