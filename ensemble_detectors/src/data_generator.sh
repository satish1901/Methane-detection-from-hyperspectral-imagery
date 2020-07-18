#! /bin/bash

echo "This scripts does all the data preprocessing..those steps include"
echo "1. Installing spectral libary"
echo "2. Running matched filter on raw data"
echo "3. Ortho-correcting matched filter output data"
echo "4. Creating sets of channels/bands and creating tiles of data to push to our detector for training"

cd ./Algorithm_1_matchfilter

echo "Installing sepctral library"
python setup.py install
echo "Finished Installing spectral library"

echo "Running MATCHED FILTER from matched_filter.py"
python matched_filter.py
echo "Finished running matched_filter.py, If you want to see the output of matched filter the output folder is created at ../../data/raw_data."

echo "Ortho-rectification of matched_filter output. Running ortho_correction.py"
python ortho_correction.py
echo "Finished running ortho_correction.py"

echo "output is generated parallel to matched_filter output data directory"

cd ../Algorithm_2_preprocess

echo "Running terrain_img_creator.py"
python terrain_img_creator.py
echo "Finished running terrain_img_creator.py"
echo "terrain image files are stored at ../../data/terrain_img_files"

echo "Running train_data_generator.py"
python train_data_generator.py
echo "Finished running train_data_generator.py"
echo "Final training data is saved at ../../data/train_data folder"
echo "Now cd to src/Algorithm_3_mrcnn"

