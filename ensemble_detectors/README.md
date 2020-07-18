### Ensemble-Detector
Ensemble detector works on raw input data (un-rectified)[Dataset B in paper]. Each datum is 432-channel/bands file. This data is first processed in sets of channels, where each set contains 50-channels/bands. Each set have an overlap of 25-channels/bands with neighbouring sets. Overview of pipeline is shown below :

<img src="./.readme_files/method_overview.png" width="600">

### Getting Started
- If using virtual environment, activate your virtual env
- #### Getting annotation data ready
 Run annotation_generator.sh. This script takes 1 argument (path to ground truth "./data/gt_data" data)
 ```
 ./annotation_generator.sh <path to ground truth data directory>
 e.g.
 (relative path) ./annotation_generator.sh '../data/gt_data'
 or
 (absolute path)[preferred]
 ./annotation_generator.sh '/home/<username>/Methane-detection-from-hyperspectral-imagery/ensemble_detectors/data/gt_data'
 ```
 
