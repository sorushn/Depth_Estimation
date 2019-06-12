import numpy as np
import cv2
import time 

LEFT_PATH = "L{:06d}.png"
RIGHT_PATH = "R{:06d}.png"

left = cv2.VideoCapture(0)
right = cv2.VideoCapture(2)


frameId = 0

# Grab both frames first, then retrieve to minimize latency between cameras
while(True):
	if not (left.grab() and right.grab()):
		print("No more frames")
		break

	_, leftFrame = left.retrieve()
	_, rightFrame = right.retrieve()
	leftFrame = cv2.flip(leftFrame,0)
	rightFrame = cv2.flip(rightFrame,0)
	cv2.imwrite(LEFT_PATH.format(frameId), leftFrame)
	cv2.imwrite(RIGHT_PATH.format(frameId), rightFrame)

	cv2.imshow('left', leftFrame)
	cv2.imshow('right', rightFrame)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break
	frameId += 1
	# time.sleep(1)

left.release()
right.release()
cv2.destroyAllWindows()