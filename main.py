import pykitti
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimization
import cv2
from matplotlib.colors import LogNorm
from visualodfunctions import *


basedir='/home/user/Downloads/data_odometry_gray/dataset_small/'
sequence = '00'
num_imgs=1000#4540
useransac=False

dataset = pykitti.odometry(basedir, sequence, frames=range(num_imgs))
f     =7.188560000000e+02  # lense focal length
baseline = dataset.calib.b_gray  # distance in m between the two cameras

cl=iter(dataset.cam0)
cr=iter(dataset.cam1)
posestrue=dataset.poses[0:num_imgs]
PM=dataset.calib.P_rect_00
points=np.zeros((1,0,2))
RTtot=np.eye(4)
posesxz=np.zeros((2, num_imgs))
posesxztrue=np.zeros((2, num_imgs))
plt.figure(figsize=(10, 10))
for i in range(num_imgs):
    right = np.array(next(cr))
    leftnew = np.array(next(cl))
    disparityL =compute_left_disparity_map(leftnew, right,
                               matcher='sgbm', verbose=True)
    if i==0:
        points=np.array(distract_keypoint(leftnew))
    if i>0:
        pointsold=np.copy(points)
        worldptsold=np.copy(worldpts)
        points,k, st=feature_tracking(left, leftnew, pointsold)
        point_in_img_mask=[(points[:,0,0]<np.shape(leftnew)[1]) & (points[:,0,1]<np.shape(leftnew)[0]) & (points[:,0,0]>0) & (points[:, 0, 1] > 0)]

        points = points[point_in_img_mask]
        worldptsold=worldptsold[point_in_img_mask]

    depthL = f / disparityL * baseline  # images are stereo rectified
    depthL[depthL > 200] = np.NaN
    depthL[depthL < 0] = np.NaN
    coordlist = list(points[:, 0, :].T.astype('int'))
    depthLpts = depthL.T[coordlist]
    points = points[np.isfinite(depthLpts)]
    if i>0:
        worldptsold=worldptsold[np.isfinite(depthLpts)]
    depthLpts = depthLpts[np.isfinite(depthLpts)]
    worldpts=screen2cam(points,depthLpts,PM)
    left=leftnew
    plt.subplot(2,2,1)
    plt.cla()
    plt.imshow(left,cmap=plt.get_cmap('gray'))
    plt.title('left image with feature points')
    plt.scatter(points[:,0,0],points[:,0,1])
    plt.subplot(2,2,2)

    plt.imshow(depthL,cmap=plt.get_cmap('gist_rainbow'),norm=LogNorm(vmin=1, vmax=200))
    plt.title('depthmap')
    plt.colorbar()
    if i>0:
        if useransac==True:
            params0=RANSAC(points, worldptsold, PM)
        else:
            pointslsq,worldptsoldlsq, worldptslsq=inlierdetection(points, worldptsold, worldpts)
            if i>1:
                x0=params0.x
            else:
                x0=np.zeros(6)
            params0 = optimization.least_squares(fun=errfct, x0=x0, args=(worldptsoldlsq,pointslsq,PM),method='lm')
        RT=createRT(params0.x[0],params0.x[1],params0.x[2],params0.x[3],params0.x[4],params0.x[5])
        RTtot=RTtot @ np.linalg.inv(RT)
        print(f'RTtot={RTtot}')
        plt.subplot(2,2,3)
        posesxz[:,i]=RTtot[[0,2],3]
        posesxztrue[:,i]=posestrue[i][[0,2],3]
        plt.cla()
        plt.plot(posesxz[0,:(i+1)],posesxz[1,:(i+1)] ,color='r')
        plt.plot(posesxztrue[0, :(i + 1)], posesxztrue[1, :(i + 1)],color='g')
        #plt.xlim([-100,100])
        #plt.ylim([-5, 195])
        plt.axis('square')
        plt.title('estimated pose (red) and true pose (green)')
    if len(points)<100:
        points = np.vstack([points,np.array(distract_keypoint(leftnew))])
        coordlist = list(points[:, 0, :].T.astype('int'))
        depthLpts = depthL.T[coordlist]
        points = points[np.isfinite(depthLpts)]
        depthLpts = depthLpts[np.isfinite(depthLpts)]
        worldpts = screen2cam(points, depthLpts, PM)
        print('detect new points')

    plt.show(block=False)
    #plt.savefig(f'imgs00c/pic{str(i).zfill(4)}.jpg')
    plt.pause(0.001)
plt.show()