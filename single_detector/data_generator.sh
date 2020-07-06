#! /bin/bash

#DATA_PATH = $1
cd ./annotation_tool
pwd

echo "Running mask_tiling.py .."
python mask_tiling.py $1
echo "Finished mask_tilling"

echo "Running json_creator.py .."
python json_creator.py $1
echo "Finished creating annotation file"

echo "Running train_data_generator.py .."
python train_data_generator.py $1
echo "Finished generating data for training"

cd ../
pwd

echo "Now cd to src/custom-mask-rcnn-detector"
echo "Run python detector.py --mode train"
echo "For more information open detector.py file and read about file usage"






