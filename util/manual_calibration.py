import cv2
from matplotlib import pyplot as plt
import numpy as np

#Measured Values
dX = 500
dY = 350
dZ = 570 #mm
dx = 337
dy = 240 #px

# Compute Intrinsics Matrix from measured values
fx = dx*dZ/dX
fy = dy*dZ/dY
K = np.diag([fx, fy, 1])
K[0, 2] = 0.5*640
K[1, 2] = 0.5*480


# def calibrate(image_size):
#     row, col = image_size
#     fx = dX*col/640
#     fy = dY*row/480
#     K = np.diag([fx, fy, 1])
#     K[0, 2] = 0.5*col
#     K[1, 2] = 0.5*row
#     return K 

def measure_object_image():
    img = cv2.imread('manual_calibration.png')
    plt.imshow(img, 'gray')
    
    # select 2 opposite corners of the rectangular object
    pts = np.asarray(plt.ginput(2, 0))
    print(pts[1]-pts[0])

def capture():
    # capture an image of a rectangular object roughly at the center of the camera's view. Press C to capture.
    cap1 = cv2.VideoCapture(2)
    while True:
        _, img = cap1.read()
        if img is None:
            continue
        img = cv2.flip(img,0)
        cv2.imshow('camera feed', img)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('c'):
            cv2.imwrite("manual_calibration.png", img)
        if c & 0xFF == ord('q'):
            break
    cap1.release()
    cv2.destroyAllWindows()