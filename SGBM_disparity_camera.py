import cv2
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt
import calib

window_size = 11
min_disp = 4
num_disp = 128
stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 50,
    speckleRange = 2,
    disp12MaxDiff = 2,
    P1 = 8*window_size**2,
    P2 = 4*8*window_size**2,
)

# stereo = cv2.StereoBM_create()
# stereo.setMinDisparity(0)
# stereo.setNumDisparities(112)
# stereo.setBlockSize(25)
# stereo.setSpeckleRange(5)
# stereo.setSpeckleWindowSize(150)
def show_undistorted():
    mtxL, distL, mtxR, distR = calib.calibrate()

    cap2 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(2)
    _, img = cap1.read()
    h,  w = img.shape[:2]

    newcameramtxL, roi = cv2.getOptimalNewCameraMatrix(mtxL, distL, (w, h), 1, (w, h))
    newcameramtxR, _ = cv2.getOptimalNewCameraMatrix(mtxR, distR, (w, h), 1, (w, h))
    while(True):
        start = time.time()
        # Capture frame-by-frame
        _, imgL = cap1.read()
        _, imgR = cap2.read()

        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        imgR = cv2.flip(imgR,0)
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgL = cv2.flip(imgL,0)
        
        imgR = cv2.undistort(imgR, mtxR, distR, None, newcameramtxR)
        imgL = cv2.undistort(imgL, mtxL, distL, None, newcameramtxL)
        x,y,w,h = roi
        imgR = imgR[y:y+h, x:x+w]
        imgL = imgL[y:y+h, x:x+w]

        disparity = stereo.compute(imgL, imgR).astype(np.float32)/16.0
        disparity = (disparity)/num_disp
        # plt.imshow(disparity,'gray')
        # plt.show()
        cv2.imshow('camera feed', np.concatenate((imgL,imgR),axis=1))
        cv2.imshow('frame',disparity)
        print(time.time()- start)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def show_undistorted_symmetric():
    mtx, dist = calib.calibrate_symmetric()

    cap2 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(2)
    _, img = cap1.read()
    h,  w = img.shape[:2]

    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    while(True):
        start = time.time()
        # Capture frame-by-frame
        _, imgL = cap1.read()
        _, imgR = cap2.read()

        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        imgR = cv2.flip(imgR,0)
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgL = cv2.flip(imgL,0)
        
        imgR = cv2.undistort(imgR, mtx, dist, None, newcameramtx)
        imgL = cv2.undistort(imgL, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        imgR = imgR[y:y+h, x:x+w]
        imgL = imgL[y:y+h, x:x+w]

        disparity = stereo.compute(imgL, imgR).astype(np.float32)/16.0
        disparity = (disparity)/num_disp
        # plt.imshow(disparity,'gray')
        # plt.show()
        cv2.imshow('camera feed', np.concatenate((imgL,imgR),axis=1))
        cv2.imshow('frame',disparity)
        print(time.time()- start)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def show_with_distortion():
    cap2 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(2)
    _, img = cap1.read()

    while(True):
        start = time.time()
        # Capture frame-by-frame
        _, imgL = cap1.read()
        _, imgR = cap2.read()

        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        imgR = cv2.flip(imgR,0)
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgL = cv2.flip(imgL,0)
        
        disparity = stereo.compute(imgL, imgR).astype(np.float32)/16.0
        disparity = (disparity)/num_disp
        # plt.imshow(disparity,'gray')
        # plt.show()
        cv2.imshow('camera feed', np.concatenate((imgL,imgR),axis=1))
        cv2.imshow('frame',disparity)
        print(time.time()- start)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--distorted", help="don't undistort the images")
    parser.add_argument("-s", "--symmetric", help="use the same calibration for both cameras")
    args = parser.parse_args()
    if args.distorted:
        show_with_distortion()
    elif args.symmetric:
        show_undistorted_symmetric()
    else:
        show_undistorted()