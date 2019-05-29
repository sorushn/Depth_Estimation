import numpy as np
import cv2
from matplotlib import pyplot as plt

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)
while(True):
    
	# Capture frame-by-frame
	#TODO: calibrate the cameras
	_, imgL = cap1.read()
	_, imgR = cap2.read()
	
	if (imgL is None) or (imgR is None):
		continue
	imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
	imgR = cv2.flip(imgR,0)
	imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
	imgL = cv2.flip(imgL,0)
	# disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
	# disparity = (disparity-min_disp)/num_disp
	# plt.imshow(disparity,'gray')
	# plt.show()
	cv2.imshow('camera feed', np.concatenate((imgL,imgR),axis=1))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap1.release()
cap2.release()
cv2.destroyAllWindows()