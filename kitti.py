import pykitti
import matplotlib.pyplot as plt
import numpy as np
import cv2

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

def createRT(phiz1,phix,phiz2,tx,ty,tz):
    R=rot_z(phiz2)@rotx(phix)@rotz(phiz1)
    RT=np.zeros((4,4))
    RT[:3,:3]=R
    RT[:,3]=np.array([[tx],[ty],[tz],[1]])
    return RT

def projectpointsback(points0,depths0,P,RT):
    #project points0 to 3d left cam system using depths and projection matrix P
    #transform to new left cam system using RT Matrix
    #project 3d points to screen in new position using P
    #return point coordinates on screen in new position
    projpoints=np.zeros_like(points0)
    for ip in range(len(points0)):
        point=points0[ip,0]
        print(f'point={point}')
        point0screen3=np.ones((3,1)) #point in camera position 0, coordinates on screen, 3 dimensional vector
        point0screen3[0:2,0]=point
        point0screen3[0,0]=point0screen3[0,0]-P[0,3] #get rid of 4th column in projection matrix, is simple addition
        point0screen3[1,0] = point0screen3[1,0] - P[1,3]
        #print(f'point0screen={point0screen3}')
        #print(f'P={P[:,:3]}')
        #print(f'P-1={np.linalg.inv(P[:, :3])}')
        point0cam3=np.linalg.inv(P[:,:3])@point0screen3 #z=1
        #print(f'point0cam3={point0cam3}')
        #print(f'Ppoint0cam3={P[:,:3]@point0cam3}')
        point0cam3=point0cam3*depths0[ip]/point0cam3[2,0]
        #print(f'point0cam3={point0cam3}')
        point0cam4=np.ones((4,1))
        point0cam4[:3]=point0cam3
        point1cam4=RT@point0cam4
        #print(f'point1cam4={point1cam4}')
        point1screen3=P@point1cam4
        point1screen3 = point1screen3/point1screen3[2,0]
        print(f'point1screen={point1screen3}')
        projpoints[ip,0,:]=point1screen3[:2,0]
    return projpoints

def dist(points0,points1):
    #calculate geometric error of all points
    return

def errfct(params,points0,depths0,points1,P):
    RT=createRT(params[0],params[1],params[2],params[3],params[4],params[5])
    points1proj=projectpointsback(points0,depths0,P,RT)
    # use for least square algorithm for optimizing RT Matrix parameters
    return dist(points0,points1proj)




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
        pointsint=np.copy(points)
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
    depthL=f/disparityL*baseline #images are stereo rectified
    depthL[depthL>200]=np.NaN
    depthL[depthL<0]=np.NaN
    Pmatrix=dataset.calib.P_rect_00
    #print(np.shape(pointsint))
    print(np.shape(depthL[list(pointsint)]))
    print(projectpointsback(points,depthL[list(pointsint)],Pmatrix,np.eye(4)))
    plt.imshow(depthL)
    plt.colorbar()
    plt.show()


