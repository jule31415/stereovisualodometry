import random
import numpy as np
import cv2
def fundemantal_matrx(keypoints1,keypoints2):
    kp1=np.float32([i for i in keypoints1])
    kp2=np.float32([i for i in keypoints2])
    A = []
    for i in range(len(keypoints1)):
        x1,y1=kp1[i]
        x2,y2=kp2[i]
        A.append([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
    U, D, V = np.linalg.svd(np.array(A))

    F = V[-1].reshape(3, 3)

    Uf, Df, Vf = np.linalg.svd(F)
    Df[2] = 0
    F = np.dot(np.dot(Uf, np.diag(Df)), Vf)
    return F

def distance(F,keypoint1,keypoint2):
    kp1=np.array([keypoint1[0],keypoint1[1],1])
    kp2=np.array([keypoint2[0],keypoint2[1],1])
    line1=F.dot(np.transpose(kp1))
    line2=np.array([-line1[1],line1[0]])
    dis=np.linalg.norm(np.cross(kp2[:], line2)/np.linalg.norm(line2))
    return dis


def ransac(keypoints1,keypoints2,iter=1000,threshold=0.1):
    F=None
    real_inliers1=[]
    real_inliers2=[]
    for i in range(iter):
        kp1=[]
        kp2=[]
        for j in range(8):
            idx=random.randint(0,len(keypoints1)-1)
            kp1.append(keypoints1[idx])
            kp2.append(keypoints2[idx])
        F=fundemantal_matrx(np.array(kp1),np.array(kp2))
        inliers1=[]
        inliers2=[]
        for k in range(len(kp1)):
            dis=distance(F,kp1[k],kp2[k])
            if (dis<threshold):
                inliers1.append(kp1[k])
                inliers2.append(kp2[k])
        if (len(inliers2)>len(real_inliers2)):
            F=fundemantal_matrx(np.array(inliers1),np.array(inliers2))
            real_inliers1=inliers1
            real_inliers2=inliers2

    return real_inliers1,real_inliers2
