import os
import time

import cv2
import numpy as np

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#Number of Rows and Columns on the Chessboard
ROWS = 9
COLS = 6

def calibrate():
    """ calculate the camera distortion parameters for each camera separately """

    #calculate for left camera
    objp = np.zeros((ROWS*COLS, 3), np.float32)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp[:, :2] = np.mgrid[0:ROWS, 0:COLS].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    path = "dataset/capture/left/"
    for fname in os.listdir(path):
        img = cv2.imread(path + fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (ROWS, COLS), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (ROWS, COLS), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(	objpoints,
                                                        # imgpoints,
                                                        # gray.shape[::-1],None,None)

    #Default calibration which includes obth radial and tangential doesn't work as expected,
    #so we need to exclude tangential distortion (using criteria argument)
    _, mtx_left, dist_left, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, 4, None, None,
        cv2.CALIB_ZERO_TANGENT_DIST+cv2.CALIB_FIX_K3,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 2e-16))

    #calculate for right camera
    objp = np.zeros((ROWS*COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:ROWS, 0:COLS].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    path = "dataset/capture/right/"
    for fname in os.listdir(path):
        img = cv2.imread(path + fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (ROWS, COLS), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (ROWS, COLS), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    _, mtx_right, dist_right, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, 4, None, None,
        cv2.CALIB_ZERO_TANGENT_DIST+cv2.CALIB_FIX_K3,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 2e-16))

    return mtx_left, dist_left, mtx_right, dist_right

def calibrate_symmetric():
    """ calculate distortion parameters for only one of the cameras """

    objp = np.zeros((ROWS*COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:ROWS, 0:COLS].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    path = "dataset/capture/left/"
    for fname in os.listdir(path):
        img = cv2.imread(path + fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (ROWS, COLS), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (ROWS, COLS), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(	objpoints,
                                                        # imgpoints,
                                                        # gray.shape[::-1],None,None)

    _, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, 4, None, None,
        cv2.CALIB_ZERO_TANGENT_DIST+cv2.CALIB_FIX_K3,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 2e-16))

    return mtx, dist


def calibrate_and_display():
    """utility function for testing the script. Displays calibrated camera output"""
    mtx, dist = calibrate_symmetric()
    capL = cv2.VideoCapture(0)
    capR = cv2.VideoCapture(2)
    _, imgL = capL.read()
    _, imgR = capR.read()
    h, w = imgL.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    while True:
        _, imgL = capL.read()
        _, imgR = capR.read()
        imgL = cv2.flip(imgL, 0)
        imgR = cv2.flip(imgR, 0)
        dstL = cv2.undistort(imgL, mtx, dist, None, newcameramtx)
        dstR = cv2.undistort(imgR, mtx, dist, None, newcameramtx)        

        # crop the image
        x, y, w, h = roi
        dstL = dstL[y:y+h, x:x+w]
        dstR = dstR[y:y+h, x:x+w]
        # cv2.imshow('l', np.array(dstL, dtype=np.uint8))
        cv2.imshow('l', np.concatenate((dstL, dstR), axis=1))
        c = cv2.waitKey(25)
        if c & 0xFF == ord('q'):
            break
        if c & 0xFF == ord('c'):
            cv2.imwrite('%f.png'%time.time(), dstL)

if __name__ == "__main__":
    calibrate_and_display()
