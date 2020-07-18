#!i /bin/bash

echo "This script will generate the training data for your detector. This will generate terrain image files and read ortho-corrected matched filter output files and create tiles of data from corresponding channels/bands set"

echo "Running terrain_img_creator.py"
python terrain_img_creator.py
echo "Finished running terrain_img_creator.py"
echo "terrain image files are stored at ../../data/terrain_img_files"

echo "Running train_data_generator.py"
python train_data_generator.py
echo "Finished running train_data_generator.py"
echo "Final training data is saved at ../../data/train_data folder"
echo "Now cd to src/Algorithm_3_mrcnn"
