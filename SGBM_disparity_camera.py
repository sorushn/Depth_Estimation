import argparse
import time

import cv2
import numpy as np

import calibration

WINDOW_SIZE = 11
MIN_DISPARITY = 4
NUM_DISPARITY = 128
# Semi-global Block Matching algorithm
stereo = cv2.StereoSGBM_create(
    minDisparity=MIN_DISPARITY,
    numDisparities=NUM_DISPARITY,
    blockSize=WINDOW_SIZE,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=2,
    disp12MaxDiff=2,
    P1=8*WINDOW_SIZE**2,
    P2=4*8*WINDOW_SIZE**2,
)

# # BlockMatching algorithm which is much faster but has poor results.
# stereo = cv2.StereoBM_create()
# stereo.setMinDisparity(0)
# stereo.setNumDisparities(112)
# stereo.setBlockSize(25)
# stereo.setSpeckleRange(5)
# stereo.setSpeckleWindowSize(150)

def show_undistorted():
    """ calculate & display disparity map produced from undistorted camera output
        which is calculated for each camera separately"""
    mtx_left, dist_left, mtx_right, dist_right = calibration.calibrate()

    cap2 = cv2.VideoCapture(2)
    cap1 = cv2.VideoCapture(0)
    _, img = cap1.read()
    h, w = img.shape[:2]

    newcameramtx_left, roi = cv2.getOptimalNewCameraMatrix(mtx_left, dist_left, (w, h), 1, (w, h))
    newcameramtx_right, _ = cv2.getOptimalNewCameraMatrix(mtx_right, dist_right, (w, h), 1, (w, h))
    while True:
        start = time.time()
        # Capture frame-by-frame
        _, left_img = cap1.read()
        _, right_img = cap2.read()

        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        right_img = cv2.flip(right_img, 0)
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        left_img = cv2.flip(left_img, 0)

        right_img = cv2.undistort(right_img, mtx_right, dist_right, None, newcameramtx_right)
        left_img = cv2.undistort(left_img, mtx_left, dist_left, None, newcameramtx_left)
        x, y, w, h = roi
        right_img = right_img[y:y+h, x:x+w]
        left_img = left_img[y:y+h, x:x+w]

        disparity = stereo.compute(left_img, right_img).astype(np.float32)/16.0
        disparity = (disparity)/NUM_DISPARITY
        # plt.imshow(disparity,'gray')
        # plt.show()
        cv2.imshow('camera feed', np.concatenate((left_img, right_img), axis=1))
        cv2.imshow('frame', disparity)
        print(time.time()- start)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def show_undistorted_symmetric():
    """ calculate & display disparity map produced from undistorted camera output
        which is calculated for one camera and applied to both """
    mtx, dist = calibration.calibrate_symmetric()

    cap2 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(2)
    _, img = cap1.read()
    h, w = img.shape[:2]

    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    while True:
        start = time.time()
        # Capture frame-by-frame
        _, left_img = cap1.read()
        _, right_img = cap2.read()

        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        right_img = cv2.flip(right_img, 0)
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        left_img = cv2.flip(left_img, 0)

        right_img = cv2.undistort(right_img, mtx, dist, None, newcameramtx)
        left_img = cv2.undistort(left_img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        right_img = right_img[y:y+h, x:x+w]
        left_img = left_img[y:y+h, x:x+w]

        disparity = stereo.compute(left_img, right_img).astype(np.float32)/16.0
        disparity = (disparity)/NUM_DISPARITY
        # plt.imshow(disparity,'gray')
        # plt.show()
        cv2.imshow('camera feed', np.concatenate((left_img, right_img), axis=1))
        cv2.imshow('frame', disparity)
        print(time.time()- start)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def show_with_distortion():
    """ calculate and display disparity map without calibrating cameras"""
    cap2 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(2)

    while True:
        start = time.time()
        # Capture frame-by-frame
        _, left_image = cap1.read()
        _, right_image = cap2.read()

        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        # right_image = cv2.flip(right_image, 0)
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        left_image = cv2.flip(left_image, 0)

        disparity = stereo.compute(left_image, right_image).astype(np.float32)/16.0
        disparity = (disparity)/NUM_DISPARITY
        # display left and right camera feed side by side
        cv2.imshow('camera feed', np.concatenate((left_image, right_image), axis=1))
        # display disparity map
        cv2.imshow('frame', disparity)
        print(time.time()- start) # show time spent on each loop
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
