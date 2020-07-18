#! /bin/bash

cd ./annotation_tool
pwd

echo "Running mask_tiling.py"
python mask_tiling.py $1
echo "Finished mask_tiling"

echo "Running json_creator.py"
python json_creator.py $1
echo "Finished creating annotation file"

cd ../

echo "Now cd to src/Algorithm_1_matchfilter directory"
