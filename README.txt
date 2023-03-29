
stereo visual odometry

This archive contains:
demo.ipynb (Demo notebook for 2 frames)
main.py (main python file that produces animation for 200 frames)
video that shows results vor 4000 frames
Documentation
a small dataset

used libs:
pykitti https://github.com/utiasSTARS/pykitti
opencv
numpy
scipy
matplotlib

used dataset:
kitti odometry dataset https://www.cvlibs.net/datasets/kitti/eval_odometry.php
use grayscale odometry dataset
download also positions and calibration files, copy the folders to the dataset directory
The download is very big (20GB), to make things easier we uploaded the data for first 200 frames together with our code.
replaced calib file in image dataset manually with calib file in calib download becaurse values were missing- need to find a better solution!

in one case, a problem with QT occured when using opencv, I solved this by
pip uninstall opencv-python
pip install opencv-python-headless
but should work normally without this change