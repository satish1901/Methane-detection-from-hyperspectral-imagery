#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:27:55 2019

@author: Satish
"""

import os
import numpy as np
import spectral as spy
import spectral.io.envi as envi
import spectral.algorithms as algo
from spectral.algorithms.detectors import MatchedFilter, matched_filter

import logging
import coloredlogs

import shutil
import statistics

# set the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aviris_data_loader")
coloredlogs.install(level='DEBUG', logger=logger)

DIRECTORY = "/media/data/satish/avng.jpl.nasa.gov/pub/test_unrect"
FILES = []
for x in os.listdir(DIRECTORY):        
    if(os.path.isdir(os.path.join(DIRECTORY, x))):
        FILES.append(x)
print(FILES)
t_sig_path = f'{DIRECTORY}/methane_signature.txt'
t_mean = np.loadtxt(t_sig_path)[:,-1]

#If target signature is not available then we can compute from the image itself to get rogh estimate.
class target_sig:
    def __init__(self):
        print("Sample class for computing target signature")
        #Future use for generalization to other gases

    #%% suplimentary function for 
    def is_safe(self, gt_image, row, col, count ,row_size, col_size, channel):
        return(row>=0 and col>=0 and row<row_size and col<col_size)

    # Facing the problem of stackoverflow in recursive implementation, iterative approach.    
    def itr_DFS(self, gt_image, row, col, count, row_size, col_size, channel, x_points, y_points):
        if((is_safe(gt_image, row, col, count, row_size, col_size, channel)) == False):
            return
        if(gt_image[row][col][channel] >= 200):
            plume_is = [(row,col,channel)]
            while plume_is:
                row,col,channel = plume_is.pop()
                gt_image[row][col][channel] = count
            
                #checking neighbours now row,col +- 1
                for delta_row in range(-1,2):
                    row_dr = row + delta_row
                    if(row_dr>=0 and row_dr<row_size):
                        for delta_col in range(-1,2):
                            col_dc = col + delta_col
                            if (delta_row == 0) ^ (delta_col == 0):
                                if(col_dc>=0 and col_dc<col_size):
                                    if(gt_image[row_dr][col_dc][channel] >= 200):
                                        plume_is.append((row_dr,col_dc,channel))
                                        x_points.append((int(row_dr)))
                                        y_points.append((int(col_dc)))

    #%%  get the coordinates of the target signature(CH4)
    def target_sign(self, GTfile_path, img_data_obj, channel, good_bands):
        print("calculating target sign from ground truth")
        source_count = 0
        norm_value = 0
        t_mean = np.zeros((channel,), dtype=np.float64)
        r_size, c_size, bands = gt_image.shape
        for channel in range((bands-1),-1,-1):
            done = False
            while not done:
                x_points = []
                y_points = []
                clrd_spot = np.transpose(np.argmax(gt_image[:,:,channel] > 200)) # [:,:,#channel]BGR
                if(clrd_spot == 0):
                    done = True
                    break
                r = clrd_spot//c_size
                c = clrd_spot%c_size
                if(is_safe(gt_image, r, c, source_count, r_size, c_size, channel)):
                    itr_DFS(gt_image, r, c, source_count, r_size, c_size, channel, x_points, y_points)
                    norm_value += len(x_points)
                    t_mean += np.array(enclosing_rect_mean(x_points, y_points, img_data_obj, good_bands)) * len(x_points)
        t_mean = np.array(t_mean) / norm_value
        print(t_mean.shape)
        print(t_mean)
        return t_mean

    #%% merge sort to keep check of indexes of y-coordinates
    def sort_coordinates(self, x_points, y_points):
        if len(x_points)>1:
            mid = len(x_points)//2
            L_x = x_points[:mid]
            L_y = y_points[:mid]
            R_x = x_points[mid:]
            R_y = y_points[mid:]
        
            sort_coordinates(L_x, L_y)
            sort_coordinates(R_x, R_y)
        
            i = j = k = 0
            while i < len(L_x) and j < len(R_x): 
                if L_x[i] < R_x[j]: 
                    x_points[k] = L_x[i]
                    y_points[k] = L_y[i]
                    i+=1    
                else: 
                    x_points[k] = R_x[j]
                    y_points[k] = R_y[j] 
                    j+=1
                k+=1
            while i < len(L_x): 
                x_points[k] = L_x[i]
                y_points[k] = L_y[i]
                i+=1
                k+=1
          
            while j < len(R_x): 
                x_points[k] = R_x[j] 
                y_points[k] = R_y[j] 
                j+=1
                k+=1
        return x_points, y_points

    #%% getting the co-ordinates range of largest rectange    
    def enclosing_rect_mean(self, x_points, y_points, img_data_obj, good_bands):
        min_x = min(x_points)
        max_x = max(x_points)
        min_y = min(y_points)
        max_y = max(y_points)
    
        x_points,y_points = np.array(x_points), np.array(y_points)
        mask_x = x_points-min_x
        mask_y = y_points-min_y

        t_img_data = img_data_obj.read_subregion((min_x,max_x), (min_y,max_y), good_bands)
        mask = np.zeros((max(mask_x+1),max(mask_y+1)))
        print("size of mask", mask.shape)
        mask[mask_x, mask_y] = 1
    
        print("Done reading target image, Calculating target_mean...")
        t_mean, t_cov, t_S = algo.mean_cov(t_img_data, mask=None, index=None)
    
        return t_mean

#%% return image object
def image_obj(hdr, img):
    "create a object of the image corresponding to certain header"
    head = envi.read_envi_header(hdr)
    param = envi.gen_params(head)
    param.filename = img   # spectral data file corresponding to .hdr file
    
    if 'bbl' in head:
        head['bbl'] = [float(b) for b in head['bbl']]
        print(len(head['bbl']))
        good_bands = np.nonzero(head['bbl'])[0]     # get good bands only for computation
        bad_bands = np.nonzero(np.array(head['bbl'])==0)[0]
        print("calculating target signature...")
        t_mean = target_sign(png_img, img_data_obj, channel, good_bands)
    
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

#%% creates a list of bands to read everytime    
def set_bands_toread(channel):
    overlap = 100; reset_pos = 0 ; num_channel = 200; start_pos=0
    overall_set = []
    band_set = []
    while(start_pos<channel):
        band_set.append(start_pos)
        start_pos+=1 ; reset_pos+=1
        if((reset_pos%num_channel)==0):
            reset_pos = 0
            overall_set.append(band_set)
            band_set = []
            start_pos = start_pos-overlap
    overall_set.append(band_set)
    return overall_set

#%% Matched Filter
for fname in FILES:
    fname_unrect = fname.split("_")[0]
    sname   = f'{fname_unrect}_rdn_v1f_clip'  #spectral data
    hname   = f'{sname}.hdr'  # header 
    dir_path= f'{DIRECTORY}/{fname}'
    img_img = f'{DIRECTORY}/{fname}/{sname}'
    hdr_img = f'{DIRECTORY}/{fname}/{hname}'
    
    row_size, col_size, channel = spy.envi.open(hdr_img).shape 
    img_data_obj = image_obj(hdr_img, img_img)
    
    if (channel-t_mean.size)>0:
        target_mean = np.append(t_mean,np.zeros((channel-t_mean.size)))
    elif (channel-t_mean.size)<0:
        target_mean = t_mean[0:channel]
        
    mf_output_folder = f'{dir_path}/{sname}_mfout'
    if not(os.path.isdir(mf_output_folder)):
        os.mkdir(mf_output_folder)
        print("\nDirectory", mf_output_folder ," created.")
    elif os.path.isdir(mf_output_folder):
        print("\nDirectory", mf_output_folder ," already exists..deleting it")
        shutil.rmtree(mf_output_folder)
        os.mkdir(mf_output_folder)
        print("\nNew Directory", mf_output_folder ," created.")
    
    #reading the whole image at once as we have a 128GB of RAM size
    print("Reading image.. ", fname)
    big_img_data = img_data_obj.read_subregion((0,row_size), (0,col_size))
    
    start_pos = 0; overlap = 100; bands_in_set = 200; reset_pos = 0
    while(start_pos<channel):
        alpha = np.zeros((row_size,0), dtype=np.float)
        '''
        read a sectiion from the image and get the background mean vector, we are
        using the push-broom aproach will overlapping matirx to calculate the mean
        covarince matrix
        '''
        end_pos = min(start_pos+bands_in_set, channel+1)
        print("taking sub-section from the big_img_data background",start_pos," to", end_pos)
        b_img_data =  big_img_data[:,:,start_pos:end_pos].copy()
        '''
        mean = vector of means of each band
        cov = bands x bands size covariance matrix
        S = number of values used on calculating mean and cov
        '''
#        b_mean, cov, S = algo.mean_cov(b_img_data, mask=None, index=None)
        for columns in range(0,b_img_data.shape[1],5):
            print("Calculating gausian stats, mean, cov of background")
            col_range = min(columns+5,b_img_data.shape[1])
            b_mean_cov_obj = algo.calc_stats(b_img_data[:,columns:col_range,:], mask=None, index=None)
            print("Calculating stats of matchFilter...")
            '''
            Matched Filter for processing hyperspectral data
            H0 background distribution -> get it from the most of the images so as to 
            have a good background distribution
            H1 target distribution -> get it from the multiple image with the source information.
        
            aplha(x)=\frac{(\mu_t-\mu_b)^T\Sigma^{-1}(x-\mu_b)}{(\mu_t-\mu_b)^T\Sigma^{-1}(\mu_t-\mu_b)}
            OR
            aplha(x) = a_hat = (transpose(x-u) . inv_cov . (t-u) / (transpose(t-u) . inv_cov . (t-u))
            '''
            print("b_img_data", b_img_data[:,columns:col_range,:].shape, "target_mean : ", \
                    target_mean[start_pos:start_pos+bands_in_set].shape) 
            alpha = np.concatenate((alpha, matched_filter(b_img_data[:,columns:col_range,:], \
                    target_mean[start_pos:start_pos+bands_in_set], b_mean_cov_obj)), axis=1)
            print("Shape of Alpha : ", alpha.shape)
        b_img_data = None
        del b_img_data
        bands_set_end = start_pos+bands_in_set
        processed_file = f'{mf_output_folder}/{sname}_{start_pos}_{end_pos}.npy'
        print("Saving the alpha output", processed_file)
        np.save(processed_file, alpha)

        start_pos+=bands_in_set; reset_pos+=bands_in_set
        if(start_pos>channel): break
        if((reset_pos%bands_in_set)==0):
            start_pos = start_pos-overlap ; reset_pos = 0

