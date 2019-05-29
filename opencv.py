import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import time
from cv2 import ximgproc
# imgR = cv2.imread("0_L.png",0)
# imgL = cv2.imread('0_R.png',0)
imgR = cv2.imread("L.ppm",0)
imgL = cv2.imread('R.ppm',0)


window_size = 11
min_disp = 4	
num_disp = 64
stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 45,
    speckleRange = 16,
    disp12MaxDiff = 1,
    P1 = 1*window_size**2,
    P2 = 4*1*window_size**2,
)

# stereo = cv2.StereoBM_create()
# stereo.setMinDisparity(min_disp)
# stereo.setNumDisparities(num_disp)
# stereo.setBlockSize(window_size)
# stereo.setSpeckleRange(10)
# stereo.setSpeckleWindowSize(40)

start = time.time()
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
disparity = (disparity-min_disp)/num_disp
end = time.time()
print("runtime:", end - start)
# filt = ximgproc.createDisparityWLSFilter(stereo)
# disparity = filt.filter(disparity, imgL, right_view=imgR)
# cv2.imshow("disparity", disparity)
plt.imshow(disparity,'gray')
plt.show()
print("hello")