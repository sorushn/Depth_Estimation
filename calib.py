import numpy as np
import cv2
import os
import time
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
ROWS = 9
COLS = 6

def calibrate():
    objp = np.zeros((ROWS*COLS,3), np.float32)
    objp[:,:2] = np.mgrid[0:ROWS,0:COLS].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    path = "dataset/capture/left/"
    for fname in os.listdir(path):
        img = cv2.imread(path + fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (ROWS, COLS), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (ROWS, COLS), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(	objpoints, 
                                                        # imgpoints,
                                                        # gray.shape[::-1],None,None)

    _, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None, 4, None, None, cv2.CALIB_ZERO_TANGENT_DIST+cv2.CALIB_FIX_K3,
                                                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 2e-16))

    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None, 4,None,None,cv2.CALIB_ZERO_TANGENT_DIST,
    #                                                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 2e-16))
    objp = np.zeros((ROWS*COLS,3), np.float32)
    objp[:,:2] = np.mgrid[0:ROWS,0:COLS].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    path = "dataset/capture/right/"
    for fname in os.listdir(path):
        img = cv2.imread(path + fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (ROWS, COLS), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (ROWS, COLS), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    
    _, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None, 4, None, None, cv2.CALIB_ZERO_TANGENT_DIST+cv2.CALIB_FIX_K3,
                                                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 2e-16))
    return mtxL, distL, mtxR, distR

def calibrate_symmetric():
    objp = np.zeros((ROWS*COLS,3), np.float32)
    objp[:,:2] = np.mgrid[0:ROWS,0:COLS].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    path = "dataset/capture/left/"
    for fname in os.listdir(path):
        img = cv2.imread(path + fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (ROWS, COLS), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (ROWS, COLS), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(	objpoints, 
                                                        # imgpoints,
                                                        # gray.shape[::-1],None,None)

    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None, 4, None, None, cv2.CALIB_ZERO_TANGENT_DIST+cv2.CALIB_FIX_K3,
                                                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 2e-16))

    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None, 4,None,None,cv2.CALIB_ZERO_TANGENT_DIST,
    #                                                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 2e-16))
    return mtx, dist


def calibrate_and_display():
    
    mtx, dist = calibrate_symmetric()
    
    cap = cv2.VideoCapture(0)
    _, img = cap.read()

    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    while(True):
        _, img = cap.read()
        img = cv2.flip(img,0)
            # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imshow('l' , np.array(dst, dtype = np.uint8 ) )
        c = cv2.waitKey(25)
        if c & 0xFF == ord('q'):
            break
        if c & 0xFF == ord('c'):
            cv2.imwrite('%f.png'%time.time(), dst)

if __name__ == "__main__":
    calibrate_and_display()
