import numpy as np
import cv2
import os

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
rows = 9
cols = 6
objp = np.zeros((rows*cols,3), np.float32)
objp[:,:2] = np.mgrid[0:rows,0:cols].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

path = "dataset/capture/left/"
for fname in os.listdir(path):
	img = cv2.imread(path + fname)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# Find the chess board corners
	ret, corners = cv2.findChessboardCorners(gray, (rows,cols),None)

	# If found, add object points, image points (after refining them)
	if ret == True:
		objpoints.append(objp)

		corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		imgpoints.append(corners2)

		# Draw and display the corners
		img = cv2.drawChessboardCorners(img, (rows,cols), corners2,ret)
		cv2.imshow('img',img)
		cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(	[i.astype('float32') for i in objpoints], 
													[i.astype('float32') for i in imgpoints],
													gray.shape[::-1],None,None)


cap = cv2.VideoCapture(0)
_, img = cap.read()

h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
while(True):
	_, img2 = cap.read()
	mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
	dst = cv2.remap(img2,mapx,mapy,cv2.INTER_LINEAR)
	
	#crop
	x,y,w,h = roi
	dst = dst[y:y+h, x:x+w]
	cv2.imshow('l' , np.array(dst, dtype = np.uint8 ) )