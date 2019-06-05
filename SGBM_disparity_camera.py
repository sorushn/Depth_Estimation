import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

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

cap2 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(2)
while(True):
	start = time.time()
	# Capture frame-by-frame
	#TODO: calibrate the cameras
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