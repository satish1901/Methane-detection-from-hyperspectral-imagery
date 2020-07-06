### Single-Detector
It works on data preprocessed by matchedfilter already, this is 4-channel data files
- Channels 0,1,2 corresponds to R,G,B
- Channel 3 corresponds for processed output from matched-filter as shown below
<img src="dataset_description.png" width="500" height="200">

### Getting Started
- If using virtual environment, activate your virtual env
- #### Mask-RCNN setup 
```
cd ./single_detector/src/custom-mask-rcnn-detector/mask_rcnn
python setup.py install
```
- #### Generating training data for our detector
```
cd ./single_detector
```
Run data_generator script, it takes path to "data" directory as input
```
./data_generator.sh '<path to data directory>'
e.g. 
(relative path) ./data_generator.sh '../data' 
or
(absolute path)[preferred]
./data_generator.sh '/home/<username>/Methane-detection-from-hyperspectral-imagery/single_detector/data'
```
- #### Training the detector
you will find your processed training data and annotation file at location 
```
Training Data :  single_detector/src/custom-mask-rcnn-detector/ch4_data/train_data
Annotations   : single_detector/src/custom-mask-rcnn-detector/ch4_data/annotation_plumes.json
```
To train the detector, Run :
```
cd ./single_detector/src/custom-mask-rcnn-detector
python detector.py --mode train
```
The corresponsing trained weight will be stored at :
```
Trained weights : ./single_detector/src/custom-mask-rcnn-detector/logs
```
- #### Testing the detector
```
python detector.py --mode predict -image <path to processed input file to detector>
```
Output is generated in the same directory
