
stereo visual odometry

In this project, a stereo visual odometry pipeline has been implemented.
Input: A sequence of stereo camera images, calibration files
Output: trajectory of the camera

This archive contains:
demo.ipynb (Demo notebook for 2 frames)
main.py (main python file that produces animation for 200 frames)
video that shows results vor 4000 frames
project report
a small dataset

programming language: python

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
The download is very big (20GB)
To make things easier we uploaded the data for first 200 frames together with our code.


