import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
frames = 0
# images = glob.glob('*.jpg')
cap = cv2.VideoCapture(0)
while(frames < 50):
	_, img = cap.read()
	img = cv2.flip(img,0)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# Find the chess board corners
	ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

	# If found, add object points, image points (after refining them)
	if ret == True:
		frames += 1
		objpoints.append(objp)

		cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		imgpoints.append(corners)

		# Draw and display the corners
		cv2.drawChessboardCorners(img, (9,6), corners,ret)
		cv2.imshow('img',img)
		cv2.waitKey(1)

cv2.destroyAllWindows()
cap.release()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

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

# mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
# dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# #crop
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imshow('calibresult',dst)
