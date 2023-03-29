import pykitti
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimization
import cv2
from matplotlib.colors import LogNorm

def rot_x(phi: float):
    return np.array([[1, 0, 0],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi), np.cos(phi)]])


def rot_y(phi: float):
    return np.array([[np.cos(phi), 0, np.sin(phi)],
                     [0, 1, 0],
                     [-np.sin(phi), 0, np.cos(phi)]])


def rot_z(phi: float):
    return np.array([[np.cos(phi), -np.sin(phi), 0],
                     [np.sin(phi), np.cos(phi), 0],
                     [0, 0, 1]])

def createRT(phix,phiy,phiz,tx,ty,tz):
    #create RT matrix from 6 parameters (3 rotation, 3 translation)
    R=rot_x(phix)@rot_y(phiy)@rot_z(phiz)
    RT=np.zeros((4,4))
    RT[:3,:3]=R
    RT[:,3]=np.array([tx,ty,tz,1])
    return RT

def screen2cam(points0, depths0, P):
    #project 2D points from screen of cam0 to 3D im coordinate system of cam0
    projpoints = np.zeros((len(points0), 3))
    for ip in range(len(points0)):
        point = points0[ip, 0]
        point0screen3 = np.ones((3, 1))  # point in camera position 0, coordinates on screen, 3 dimensional vector
        point0screen3[0:2, 0] = point
        point0screen3[0, 0] = point0screen3[0, 0] - P[0, 3]  # get rid of 4th column in projection matrix, is simple addition
        point0screen3[1, 0] = point0screen3[1, 0] - P[1, 3]
        point0cam3 = np.linalg.inv(P[:, :3]) @ point0screen3  # z=1
        point0cam3 = point0cam3 * depths0[ip] / point0cam3[2, 0]
        projpoints[ip, :] = point0cam3[:, 0]
    return projpoints


def projectpointsback(points0cam3,P,RT):
    #transform 3D points to new left cam system using RT Matrix
    #project 3d points to screen in new position using P
    #return point coordinates on screen in new position
    projpoints=np.zeros((len(points0cam3),1,2))
    for ip in range(len(points0cam3)):
        point0cam3=points0cam3[ip,:]
        point0cam4=np.ones((4,1))
        point0cam4[:3,0]=point0cam3
        point1cam4=RT@point0cam4
        point1screen3=P@point1cam4
        point1screen3 = point1screen3/point1screen3[2,0]
        projpoints[ip,0,:]=point1screen3[:2,0]
    return projpoints

def dist(points0,points1):
    #calculate geometric error of all points
    return np.linalg.norm(points1-points0,axis=2)[:,0]

def errfct(params,worldpts0,points1,P):
    RT=createRT(params[0],params[1],params[2],params[3],params[4],params[5])
    points1proj=projectpointsback(worldpts0,P,RT) #L[list(pointsint)]
    # use for least square algorithm for optimizing RT Matrix parameters
    return dist(points1,points1proj)


def compute_left_disparity_map(img_left, img_right, matcher='bm', verbose=True):
    # parameters for opencv disparity functions
    sad_window = 6  # sum of absolutes (images)
    num_disparities = sad_window * 16
    block_size = 11
    matcher_name = matcher

    if matcher_name == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=num_disparities,
                                      blockSize=block_size)

    elif matcher_name == 'sgbm':
        matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1=8 * 1 * block_size ** 2,
                                        P2=32 * 1 * block_size ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    disp_left = matcher.compute(img_left, img_right).astype(np.float32) / 16


    #if verbose:
        #print(f'Time to compute disparity map using Stereo{matcher_name.upper()}', end - start)

    return disp_left

def distract_keypoint(gray):
    #detect keypoints using SIFT
    sift=cv2.SIFT_create()
    kps=sift.detect(gray,None)
    kps_bucket,coor_bucket=keypoint_bucket(kps)
    #cv2.drawKeypoints(image=gray, keypoints=kps_bucket, outImage=gray)#, color=(0, 255, 0))

    return coor_bucket

def keypoint_bucket(kps):
    #divide image in several parts and use only the 5 best keypoints of each
    keypoint =[]
    kp_coordinate=[[] for i in range(180)]
    bucket=[[] for i in range(180)]
    bucketsize_x=220
    bucketsize_y=40
    for kp in kps:
        x,y=kp.pt
        index=int(x / bucketsize_x) + int(y / bucketsize_y) *int (1762 / 220 + 1)
        bucket[index].append(kp)
    for i in range(0,180):
        if(bucket[i]!=None):
            sortedbucket=sorted(bucket[i],key=lambda kp:kp.response,reverse=True)[:5]
            for kp in sortedbucket:
                x,y=kp.pt
                kp_coordinate[i].append([[x,y]])
            keypoint.append(sortedbucket)
    result_kp = [item for sub in keypoint for item in sub]
    result_coor = [item for sub in kp_coordinate for item in sub]
    return result_kp,result_coor


def feature_tracking(gray_t1,gray_t2,kpts):
    #optical flow
    KL_param=dict(winSize=(15,15),
                  maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    p0 = kpts.astype('float32')
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray_t1, gray_t2, p0, None, **KL_param)
    k=0
    for i in st:
        if(i==1):
            k+=1
    return p1, k, st


def inlierdetection(points1, worldpoints0, worldpoints1):
    #find a subset of the 3D points of the 2 frames that are consistent in their distance to each other
    #distance betweeen 3D points should be constant even if coordinate system changes
    #use clique algorithm
    dmax=0.2
    wp0a=worldpoints0[None,:,:]
    wp0b = worldpoints0[:,None, :]
    wp1a = worldpoints1[None,:, :]
    wp1b=worldpoints1[:,None,:]
    d0=np.linalg.norm(wp0a-wp0b,axis=2)
    d1 = np.linalg.norm(wp1a - wp1b, axis=2)
    consistent=np.zeros_like(d0)
    consistent[np.abs(d1-d0)<dmax]=1
    bestpoint=np.argmax(np.sum(consistent, axis=0))
    cliqueind=[bestpoint]
    while True:
        consistentwithclique=(np.sum(consistent[cliqueind,:],axis=0)==len(cliqueind))
        consistentwithclique[cliqueind]=False
        consistentwithcliqueind=np.where(consistentwithclique)[0]
        nconsistentwithpotentialinliers = np.sum(consistent[consistentwithcliqueind, :], axis=0)
        nconsistentwithpotentialinliers[consistentwithclique==False]=0
        newcliqueind=np.argmax(nconsistentwithpotentialinliers)
        if nconsistentwithpotentialinliers[newcliqueind]<2:
            break
        cliqueind.append(newcliqueind)
        if len(cliqueind)>70:
            break


    return points1[cliqueind], worldpoints0[cliqueind], worldpoints1[cliqueind]

def RANSAC(points1,worldpoints0,PM):
    #alternative to clique
    #use subsets of 6 points each, calculate RT using least squares, find best subset using RANSAC
    real_inliers_wp0=[]
    real_inliers_p1=[]
    threshold=1.0
    for i in range(200):
        idx=np.random.randint(0,len(worldpoints0)-1,6)
        wp0=worldpoints0[idx]
        p1=points1[idx]
        params0 = optimization.least_squares(fun=errfct, x0=np.zeros(6), args=(wp0, p1, PM), method='lm',max_nfev=200)
        inlier_wp0=[]
        inliers2=[]
        dis = errfct(params0.x, worldpoints0, points1, PM)
        inliers_wp0=worldpoints0[dis<threshold]
        inliers_p1 = points1[dis < threshold]
        if (len(inliers_p1)>len(real_inliers_p1)):
            real_inliers_p1=inliers_p1
            real_inliers_wp0=inliers_wp0
            bestparams=params0
    return bestparams