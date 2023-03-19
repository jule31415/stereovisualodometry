import cv2
import numpy as np
def distract_keypoint(img):

    #img=cv2.imread('data/image_L/2018-07-11-14-48-52_2018-07-11-15-09-57-367.png')
    gray=cv2.cvtColor(src=img,code=cv2.COLOR_BGR2GRAY)
    sift=cv2.SIFT_create()
    kps=sift.detect(gray,None)
    kps_bucket,coor_bucket=keypoint_bucket(kps)
    cv2.drawKeypoints(image=gray, keypoints=kps_bucket, outImage=img, color=(0, 255, 0))
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
        bucket[index].append(kp)
    for i in range(0,180):
        if(bucket[i]!=None):
            sortedbucket=sorted(bucket[i],key=lambda kp:kp.response,reverse=True)[:10]
            for kp in sortedbucket:
                x,y=kp.pt
                kp_coordinate[i].append([[x,y]])
            keypoint.append(sortedbucket)
    result_kp = [item for sub in keypoint for item in sub]
    result_coor = [item for sub in kp_coordinate for item in sub]
    return result_kp,result_coor


def feature_tracking():
    #optical flow
    img=cv2.imread('data/image_L/2018-07-11-14-48-52_2018-07-11-15-09-57-367.png')
    gray_t1 = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    KL_param=dict(winSize=(15,15),
                  maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    #p0=cv2.goodFeaturesToTrack(gray_t1, mask=None, **feature_params)
    p0=np.array(distract_keypoint(img),dtype=np.float32)
    print(type(p0))
    #while(True):
    img2=cv2.imread('data/image_L/2018-07-11-14-48-52_2018-07-11-15-09-57-567.png')
    gray_t2 = cv2.cvtColor(src=img2, code=cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray_t1, gray_t2, p0, None, **KL_param)
    k=0
    for i in st:
        if(i==1):
            k+=1
    return k

print(feature_tracking())
