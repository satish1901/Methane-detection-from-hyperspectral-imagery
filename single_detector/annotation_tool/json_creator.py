#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#file usage : python json_creator.py ../data
"""
Created on Tue Apr  2 17:19:37 2019

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

FP_REMOVAL = True
DIRECTORY = "{}/".format(sys.argv[1])
FILES = [x for x in os.listdir(DIRECTORY)]
print(FILES)

## supplimentary functions for DFS
def is_safe(row,col, count ,row_size, col_size, channel):
    return(row>=0 and col>=0 and row<row_size and col<col_size)

def num_of_sources(row, col, count, row_size, col_size, channel, x_points, y_points):
    itr_DFS(row, col, count, row_size, col_size, channel, x_points, y_points)
    return

# iterative way of doing the DFS search to get the number of sources
# Facing the problem of stack overflow in recursive implementation, iterative approach.
    
def itr_DFS(row, col, count, row_size, col_size, channel, x_points, y_points):
    if((is_safe(row, col, count, row_size, col_size, channel)) == False):
        return
    if(np_image[row][col][channel] >= 200):
        plume_is = [(row,col,channel)]
        while plume_is:
            row,col,channel = plume_is.pop()
            np_image[row][col][channel] = count
            
            #checking neighbours now row,col +- 1
            for delta_row in range(-1,2):
                row_dr = row + delta_row
                if(row_dr>=0 and row_dr<row_size):
                    for delta_col in range(-1,2):
                        col_dc = col + delta_col
                        if (delta_row == 0) ^ (delta_col == 0):
                            if(col_dc>=0 and col_dc<col_size):
                                if(np_image[row_dr][col_dc][channel] >= 200):
                                    plume_is.append((row_dr,col_dc,channel))
                                    x_points.append((int(row_dr)))
                                    y_points.append((int(col_dc)))

#%% creating json file and dumping data(the mask information) in json format
import json
import copy

sample = {
  "default_name.png":
    {
      "filename":"default_name.png",
      "size":-1,
      "regions":[
              {
                "shape_attributes":
                  {
                    "name":"polygon",
                    "all_points_x":[],
                    "all_points_y":[]
                  },
              "region_attributes":
                {
                  "name":"default"
                }
            }
      ]
    }
}                  
            
#%% creating the dictionary for the json file
def new_image_add_json(img_name, coordinates):
    if img_name not in coordinates:
        value = dict(copy.deepcopy(sample["default_name.png"]))
        coordinates[img_name] = value
        coordinates[img_name]["filename"] = img_name
    else:
        print("do something")
    return
    
def new_region_add_json(img_name, coordinates, x_points, y_points, channel):
    if img_name in coordinates:
        region = dict(copy.deepcopy(sample["default_name.png"]["regions"][0]))
        region['shape_attributes']['all_points_x'] = x_points
        region['shape_attributes']['all_points_y'] = y_points
        if (channel == 2):
            region['region_attributes']['name'] = 'point_source'
        if (channel == 0):
            region['region_attributes']['name'] = 'diffused_source'
        if(len(x_points) == 0):
            region['region_attributes']['name'] = 'no_methane'
        coordinates[img_name]["regions"].append(region)
        if (coordinates[img_name]['regions'][0]['region_attributes']['name'] == 'default'):
            print("deleting default")
            del coordinates[img_name]['regions'][0]
    return

#%% creating json file to dump the data
def json_generate(img_name, coordinates, json_file):
    if not os.path.isfile(json_file):
        print("file doesnot exists", json_file)
        with open(json_file, mode='w') as f:    
		#if file do not exists it will create one and write data to it
            f.write(json.dumps(coordinates, indent=4))
    else:
        print("file exists", json_file)
        with open(json_file) as feedsjson:  #if json file is already there
            feeds = json.load(feedsjson)
    
        feeds.update(coordinates)
        with open(json_file, mode='w') as f:
            f.write(json.dumps(feeds, indent=4))
    return

#%% calculating the number of point and diffused sources
json_file = "../src/custom-mask-rcnn-detector/ch4_data/annotation_plumes.json"

for fname in FILES:
    print(fname)
    file_check = f'{DIRECTORY}/{fname}/{fname}_img_mask.png'
    if((os.path.isfile(file_check)) == False):
        continue
    sname = f'{fname}_img'  #spectral data
    hname = f'{sname}.hdr'  # header
    mname = f'{sname}_mask.png'  # manual annotated mask
    png_img = f'{DIRECTORY}/{fname}/{mname}'
    img_img = f'{DIRECTORY}/{fname}/{sname}'
    hdr_file = f'{DIRECTORY}/{fname}/{hname}'
    png_read = cv2.imread(png_img)
    img_shape = png_read.shape
    tile_size = (1024, 1024)
    offset = (512, 512)

    for i in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
        for j in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
            png_mask_file = f'{DIRECTORY}/{fname}/{fname}_img_mask_tiles/{fname}_img_img_radiance_{i}_{j}.png'
            png_mask_tile_name = f'{sname}_img_radiance_{i}_{j}.npy' #tilename to be written in json file

            if os.path.isfile(png_mask_file):
                print(png_mask_file)
                png_mask_read = cv2.imread(png_mask_file)
                np_image = np.array(png_mask_read)
                row_size, col_size, bands = np_image.shape
                print(row_size, col_size, bands)
                point_source = 0 #number of pointer sources
                diffused_source = 0 #number of diffused sources
                source_count = 0
                coordinates = {} # dictionary for the json file carying the information
              
                new_image_add_json(png_mask_tile_name, coordinates)

                for channel in range((bands-1),-1,-1):
                    source_count = 0
                    done = False
                    while not done:
                        x_points = []
                        y_points = []
                        colored_spots = np.transpose(np.argmax(np_image[:,:,channel] > 200)) # [:,:,#channel]BGR
                        if(colored_spots == 0):

                            done = True
                            break
                            
                        source_count = source_count + 1
                        row = colored_spots//col_size
                        col = colored_spots%col_size
                        if(is_safe(row, col, source_count, row_size, col_size, channel)):
                            num_of_sources(row, col, source_count,  row_size, col_size, channel, x_points, y_points)
                        
                        if(len(x_points) > 0):
                            new_region_add_json(png_mask_tile_name, coordinates, x_points, y_points, channel)
                            json_generate(png_mask_tile_name, coordinates, json_file) #dumping everything to the json file
                    if (channel == 2):
                        point_source = source_count
                    elif(channel == 0):
                        diffused_source = source_count
                    
                print("\npoint_source = ", point_source)
                print("\ndiffused_source = ", diffused_source)
