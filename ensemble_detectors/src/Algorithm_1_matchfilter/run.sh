#! /bin/bash

echo "This files runs Algorithm 1 as mentioned in the paper"

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

echo "Now to prepara data for training cd to src/Algorithm_2_preprocess"

