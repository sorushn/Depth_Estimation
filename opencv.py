import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from cv2 import ximgproc
imgR = cv2.imread("0_L.png",0)
imgL = cv2.imread('0_R.png',0)

# stereo = cv2.StereoBM_create(numDisparities=32, blockSize=5)
# disparity = stereo.compute(imgL,imgR)
# plt.imshow(disparity,'gray')
# plt.show()

window_size = 7
min_disp = 2
num_disp = 80
stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 150,
    speckleRange = 15,
    disp12MaxDiff = 10,
    P1 = 8*window_size**2,
    P2 = 32*window_size**2,
)
start = time.time()
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
disparity = (disparity-min_disp)/num_disp
end = time.time()
print("runtime:", end - start)
# filt = ximgproc.createDisparityWLSFilter(stereo)
# disparity = filt.filter(disparity, imgL, right_view=imgR)
plt.imshow(disparity,'gray')
plt.show()
