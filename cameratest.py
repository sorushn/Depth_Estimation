import numpy as np
import cv2
from matplotlib import pyplot as plt

window_size = 5
min_disp = 2	
num_disp = 32
stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 15,
    disp12MaxDiff = 10,
    P1 = 8*window_size**2,
    P2 = 32*window_size**2,
)
cap1 = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(1)
while(True):
    
	# Capture frame-by-frame
	#TODO: calibrate the cameras
	_, imgL = cap1.read()
	_, imgR = cap2.read()

	imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
	imgR = cv2.flip(imgR,0)
	imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
	imgL = cv2.flip(imgL,0)
	disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
	disparity = (disparity-min_disp)/num_disp
	# plt.imshow(disparity,'gray')
	# plt.show()
	cv2.imshow('frame',disparity)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap1.release()
cap2.release()
cv2.destroyAllWindows()