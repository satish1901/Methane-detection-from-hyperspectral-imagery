#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#file usage : python mask_tiling.py ../data/gt_data
"""
Created on Tue Apr  2 13:27:27 2019

@author: satish
"""

import sys
import os
import cv2
import math
import numpy as np
import logging
import coloredlogs

# set the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aviris_data_loader")
coloredlogs.install(level='DEBUG', logger=logger)

cur_dir = os.getcwd()
print(cur_dir)
DIRECTORY = "{}/".format(sys.argv[1])
print(DIRECTORY)
FILES = [x for x in os.listdir(DIRECTORY)]
print(FILES)

#%% creating tiles of the mask (i.e. mask.png)
import shutil

for fname in FILES:
    file_check = f'{DIRECTORY}/{fname}/{fname}_img_mask.png'
    if((os.path.isfile(file_check)) == False):
        continue
    sname = f'{fname}_img'  #spectral data
    hname = f'{sname}.hdr'  # header
    mname = f'{sname}_mask.png'  # manual annotated mask
    png_img = f'{DIRECTORY}/{fname}/{mname}'
    img_img = f'{DIRECTORY}/{fname}/{sname}'
    hdr_file = f'{DIRECTORY}/{fname}/{hname}'
    
    
    #if the mask exits then we will make tiles of the image and save in folder created "{name}_titles
    tiles_folder = f'{DIRECTORY}/{fname}/{fname}_img_mask_tiles'
    if not(os.path.isdir(tiles_folder)):
        print("something")
        os.mkdir(tiles_folder)
        print("\nDirectory", tiles_folder ," created.")
    elif os.path.isdir(tiles_folder):
        print("\nDirectory", tiles_folder ," already exists..deleting it")
        shutil.rmtree(tiles_folder)
        os.mkdir(tiles_folder)
        print("\nNew Directory", tiles_folder ," created.")
    print(png_img)
    img = cv2.imread(png_img)
    img_shape = img.shape
    print("shape of mask", img_shape)
    tile_size = (256, 256)
    offset = (128, 128)
    img_name = f'{sname}_img'

    
    for i in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
        for j in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
            cropped_img = img[offset[1]*i:min(offset[1]*i+tile_size[1], \
                              img_shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], \
                                       img_shape[1])]
            # Debugging the tiles
            tile_name = f'{img_name}_radiance'
            check_tile = f'{tile_name}_{i}_{j}.png'
            if(check_tile not in tiles_folder):
                cv2.imwrite(os.path.join(tiles_folder, (tile_name + "_" + str(i) \
                + "_" + str(j) + ".png")), cropped_img)
                
    print("Done Tilling mask file", mname)
