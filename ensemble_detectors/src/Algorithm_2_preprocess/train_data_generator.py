#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:50:48 2019

@author: Satish
"""

import os
import cv2
import math
import numpy as np
import spectral as spy
import spectral.io.envi as envi
import shutil

#DIRECTORY = "/media/data/satish/avng.jpl.nasa.gov/pub/test_unrect"
DIRECTORY = "../../data/raw_data"
TERRAIN_IMG = "../../data/terrain_img_files"
PROCESSEDDATA_PATH = "../../data/train_data"

FILES = []
for x in os.listdir(DIRECTORY):        
    if(os.path.isdir(os.path.join(DIRECTORY, x))):
        FILES.append(x)
print(FILES)

def ignore_and_range(MAT, k=-9999.0):
    #ignore k , normalize and set range to (0,255)
    MAT[MAT<0]=0
    MAT = (MAT-MAT.min()) / (MAT.max()-MAT.min())
    MAT *= 255
    
    return MAT

GLOBAL_RAD = {
    "min": -10.092653964613802, 
    "max": 8.085993906027412
    } 

for fname in FILES:
    rect_folder_name = fname.split("_")[0]
    fname_path = f'{DIRECTORY}/{fname}/{rect_folder_name}_rect'
    try:
        rect_fname_list = os.listdir(fname_path)
    except:
        print("File" ,fname_path," not found,skipping it, probably does not exist in manual_offset list")
        continue
    for rect_fname in rect_fname_list:
        print(rect_fname)
        rect_fname_path = f'{fname_path}/{rect_fname}'
        np_file = np.load(rect_fname_path)        
        #global normalization
        np_file_global = np_file/GLOBAL_RAD['max']
        #local image-wise normalization
        np_file_local = np_file/np_file.max()
        #loading the gray-scale terrain image
        tname = fname.split('_')[0]
        terrain_path = f'{TERRAIN_IMG}/{tname}_cmf_v1f/{tname}_cmf_v1f_img.npy'
        np_terrain = np.load(terrain_path)
        np_terrain_norm = ignore_and_range(np_terrain)
        img_mrcnn = np.dstack((np_terrain_norm, np_file_local, np_file_global))

        #creating tiles folder and saving the whole file in tiles
        tfname = (rect_fname.split('.')[0]).split('_')
        tiles_set_folder = f'{PROCESSEDDATA_PATH}/{tfname[-2]}-{tfname[-1]}'
        if not(os.path.isdir(tiles_set_folder)):
            os.mkdir(tiles_set_folder)
        else: #os.path.isdir(tiles_set_folder):
            print("\nDirectory", tiles_set_folder ," already exists")
        
        tiles_folder = f'{tiles_set_folder}/{tname}_rdn_tiles'
        if not(os.path.isdir(tiles_folder)):
            os.mkdir(tiles_folder)
        elif os.path.isdir(tiles_folder):
            shutil.rmtree(tiles_folder)
            os.mkdir(tiles_folder)
            print("\nNew Directory", tiles_folder ," created.")
        
        img_shape = img_mrcnn.shape
        tile_size = (256, 256)
        offset = (128, 128)
        img_name = f'{tname}_rdn'
                   
        for i in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
            for j in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
                cropped_tile = img_mrcnn[offset[1]*i:min(offset[1]*i+tile_size[1], \
                                  img_shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], \
                                           img_shape[1])]
    
                check_tile = f'{img_name}_{i}_{j}.npy'
                
                if(check_tile not in tiles_folder):
                    np.save(os.path.join(tiles_folder, (img_name + "_" + str(i) \
                    + "_" + str(j))), cropped_tile)
   
