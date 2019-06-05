import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import time
from cv2 import ximgproc
# imgR = cv2.imread("0_L.png",0)
# imgL = cv2.imread('0_R.png',0)
imgR = cv2.imread("dataset/0_L.png",0)
imgL = cv2.imread('dataset/0_R.png',0)


window_size = 7
min_disp = 1	
num_disp = 112
stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 2,
    speckleWindowSize = 50,
    speckleRange = 1,
    disp12MaxDiff = 5,
    P1 = 8*window_size**2,
    P2 = 4*8*window_size**2,
)

start = time.time()
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
disparity = (disparity-min_disp)/num_disp
end = time.time()
print("runtime:", end - start)
plt.imshow(disparity,'gray')
plt.show()