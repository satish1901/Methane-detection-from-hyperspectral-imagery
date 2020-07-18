### Ensemble-Detector
Ensemble detector works on raw input data (un-rectified)[Dataset B in paper]. Each datum is 432-channel/bands file. This data is first processed in sets of channels, where each set contains 50-channels/bands. Each set have an overlap of 25-channels/bands with neighbouring sets. Overview of pipeline is shown below :

<img src="./.readme_files/method_overview.png" width="600">

### Getting Started
- If using virtual environment, activate your virtual env
- #### Getting annotation data ready
```
cd ./ensemble_detectors
```
 
 Run annotation_generator.sh. This script takes 1 argument (path to ground truth "./data/gt_data" data)
 
 ```
 ./annotation_generator.sh <path to ground truth data directory>
 e.g.
 (relative path) ./annotation_generator.sh '../data/gt_data'
 or
 (absolute path)[preferred]
 ./annotation_generator.sh '/home/<username>/Methane-detection-from-hyperspectral-imagery/ensemble_detectors/data/gt_data'
 
 Annotation file saved at : Methane-detection-from-hyperspectral-imagery/ensemble_detectors/src/gt_jsonfile
 ```

- #### Generating  training data for our detector
This repo follows the same code execution flow as mentioned in the paper
1. Algorithm_1_matchfilter : Algorithm 1 in paper. It runs matched filer on the raw hyperspectral data files and then runs ortho correction to rectify the data
2. Algorithm_2_preprocess : Algorithm 2 in paper. It create training data for the detection, terrain_img_creator.py file reads the 0,1,2 bands from the 4-bands data and creates terrain image files. train_data_generator create sets of bands and creates tiles of images to be used in training. Each tile containes 3 bands :- 0-gray scale terrain image, 1-locally normalized output of matched filter, 2-globally normalized out of matched filter
3. Algorithm_3_mrcnn : Algorithm 3 in paper. It contains our detector network. detector.py is our training/testing file for network

```
cd ./src
```
Run data_generator script.
```
./data_generator.sh
```
Running time of this script depends on the size of data. It does cumulative work on Algorithm_1_matchfilter & Algorithm_2_preprocess.
Each hyperspectral image of dim : 22,000 x 1500 x 432 takes ~15 minutes of time to generate output

- #### Training the ensemble of detector
you will find your processed training data and annotation file at below location
```
Training Data : ensemble_detectors/data/train_data
Annotations   : ensemble_detectors/src/gt_jsonfile
```
##### Setup detector code
```
cd ./ensemble_detectors/src/Algorithm_3_mrcnn/mask_rcnn
python setup.py install
```



