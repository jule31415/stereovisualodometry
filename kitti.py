import pykitti
import matplotlib.pyplot as plt
import numpy as np
import cv2
basedir='/home/user/Downloads/data_odometry_gray/dataset/'
sequence = '00'
num_imgs=10
dataset = pykitti.odometry(basedir, sequence, frames=range(num_imgs))
print('test')
second_pose = dataset.poses[1]
#first_gray = next(iter(dataset.gray))
#first_cam2 = dataset.get_cam2(0)
#third_velo = dataset.get_velo(2)

f     =7.188560000000e+02  # lense focal length
baseline = dataset.calib.b_gray  # distance in m between the two cameras
#left_matcher = cv2.StereoBM_create(numDisparities=disparities, blockSize=block)
left_matcher = cv2.StereoSGBM_create(minDisparity=0,numDisparities=16, blockSize=16, P1=0, P2=0)
cl=iter(dataset.cam1)
cr=iter(dataset.cam0)
points=np.zeros((1,0,2))
for i in range(num_imgs):
    right = np.array(next(cr))
    leftnew = np.array(next(cr))
    disparityL = left_matcher.compute(leftnew, right)
    if i==0:
        dst = cv2.cornerHarris(leftnew,2,3,0.04)
        points=np.where(dst>0.01*dst.max())
        points=np.array(points).T[:,None,:].astype('float32')
    if i>0:
        points, status, err = cv2.calcOpticalFlowPyrLK(left.T, leftnew.T, points, None) #idk if left or left.T
    points=points[points[:,0,0]<np.shape(leftnew)[0]]
    points=points[points[:,0,1]<np.shape(leftnew)[1]]
    points=points[points[:,0,0]>0]
    points=points[points[:,0,1]>0]
    left=leftnew
    plt.subplot(2,1,1)
    plt.imshow(left)
    plt.colorbar()
    plt.scatter(points[:,0,1],points[:,0,0])
    plt.subplot(2,1,2)
    depthL=f/disparityL*baseline
    depthL[depthL>200]=np.NaN
    depthL[depthL<0]=np.NaN
    plt.imshow(depthL)
    plt.colorbar()
    plt.show()


