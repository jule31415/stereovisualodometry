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

    R=rot_x(phix)@rot_y(phiy)@rot_z(phiz)
    RT=np.zeros((4,4))
    RT[:3,:3]=R
    RT[:,3]=np.array([tx,ty,tz,1])
    return RT

def screen2cam(points0, depths0, P):
    projpoints = np.zeros((len(points0), 3))
    for ip in range(len(points0)):
        point = points0[ip, 0]
        # print(f'point={point}')
        point0screen3 = np.ones((3, 1))  # point in camera position 0, coordinates on screen, 3 dimensional vector
        point0screen3[0:2, 0] = point
        point0screen3[0, 0] = point0screen3[0, 0] - P[
            0, 3]  # get rid of 4th column in projection matrix, is simple addition
        point0screen3[1, 0] = point0screen3[1, 0] - P[1, 3]
        # print(f'point0screen={point0screen3}')
        # print(f'P={P[:,:3]}')
        # print(f'P-1={np.linalg.inv(P[:, :3])}')
        point0cam3 = np.linalg.inv(P[:, :3]) @ point0screen3  # z=1
        # print(f'point0cam3={point0cam3}')
        # print(f'Ppoint0cam3={P[:,:3]@point0cam3}')
        # print(f'depths={depths0}')
        point0cam3 = point0cam3 * depths0[ip] / point0cam3[2, 0]
        projpoints[ip, :] = point0cam3[:, 0]
    return projpoints


def projectpointsback(points0cam3,P,RT):
    #project points0 to 3d left cam system using depths and projection matrix P
    #transform to new left cam system using RT Matrix
    #project 3d points to screen in new position using P
    #return point coordinates on screen in new position
    projpoints=np.zeros((len(points0cam3),1,2))
    #points0cam3=screen2cam(points0, depths0, P)
    for ip in range(len(points0cam3)):
        point0cam3=points0cam3[ip,:]
        point0cam4=np.ones((4,1))
        point0cam4[:3,0]=point0cam3
        point1cam4=RT@point0cam4
        #print(f'point1cam4={point1cam4}')
        point1screen3=P@point1cam4
        point1screen3 = point1screen3/point1screen3[2,0]
        #print(f'point1screen={point1screen3}')
        projpoints[ip,0,:]=point1screen3[:2,0]
    return projpoints

def dist(points0,points1):
    #calculate geometric error of all points
    return np.linalg.norm(points1-points0,axis=2)[:,0]

def errfct(params,worldpts0,points1,P):
    RT=createRT(params[0],params[1],params[2],params[3],params[4],params[5])
    points1proj=projectpointsback(worldpts0,P,RT) #L[list(pointsint)]
    # use for least square algorithm for optimizing RT Matrix parameters
    #print(points1proj[:50])
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

    #img=cv2.imread('data/image_L/2018-07-11-14-48-52_2018-07-11-15-09-57-367.png')
    #gray=cv2.cvtColor(src=img,code=cv2.COLOR_BGR2GRAY)
    sift=cv2.SIFT_create()
    kps=sift.detect(gray,None)
    kps_bucket,coor_bucket=keypoint_bucket(kps)
    cv2.drawKeypoints(image=gray, keypoints=kps_bucket, outImage=gray)#, color=(0, 255, 0))
    #cv2.imshow('img', gray) #cv2.
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return coor_bucket

def keypoint_bucket(kps):
    keypoint =[]
    kp_coordinate=[[] for i in range(180)]
    bucket=[[] for i in range(180)]
    bucketsize_x=220
    bucketsize_y=40
    for kp in kps:
        x,y=kp.pt
        index=int(x / bucketsize_x) + int(y / bucketsize_y) *int (1762 / 220 + 1)
        #print(index)
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
    #img=cv2.imread('data/image_L/2018-07-11-14-48-52_2018-07-11-15-09-57-367.png')
    #gray_t1 = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    KL_param=dict(winSize=(15,15),
                  maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    #p0=cv2.goodFeaturesToTrack(gray_t1, mask=None, **feature_params)
    #p0=np.array(distract_keypoint(gray_t1),dtype=np.float32)
    p0 = kpts.astype('float32')
    #while(True):
    #img2=cv2.imread('data/image_L/2018-07-11-14-48-52_2018-07-11-15-09-57-567.png')
    #gray_t2 = cv2.cvtColor(src=img2, code=cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray_t1, gray_t2, p0, None, **KL_param)
    k=0
    for i in st:
        if(i==1):
            k+=1
    return p1, k, st


def inlierdetection(points1, worldpoints0, worldpoints1):
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

basedir='/home/user/Downloads/data_odometry_gray/dataset/'
sequence = '00'
num_imgs=4540
dataset = pykitti.odometry(basedir, sequence, frames=range(num_imgs))
second_pose = dataset.poses[1]
useransac=False

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
anglesum=0
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
            #bounds = (
            #    [-np.pi / 8, -np.pi / 4, -np.pi / 8, -1.0, -0.5, -1.5],
            #    [np.pi / 8, np.pi / 4, np.pi / 8, 1.0, 0.5, 1.5])
            if i>1:
                x0=params0.x
            else:
                x0=np.zeros(6)
            params0 = optimization.least_squares(fun=errfct, x0=x0, args=(worldptsoldlsq,pointslsq,PM),method='lm')
        RT=createRT(params0.x[0],params0.x[1],params0.x[2],params0.x[3],params0.x[4],params0.x[5])
        RTtot=RTtot @ np.linalg.inv(RT)
        print(f'RTtot={RTtot}')
        #print(f'RTstep={params0.x}')
        #print(f'RTstep={np.linalg.inv(RT)}')
        anglesum=anglesum+params0.x[1]
        #print(anglesum)
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
    plt.pause(0.0001)
plt.show()



