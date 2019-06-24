import numpy as np
from numpy.linalg import matrix_power
from PIL import Image
import cv2
import time
from matplotlib import pyplot as plt

AVG_FILTER_SIZE = 11
DEFAULT_PATCH_SIZE = 7
SCALE = 4

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

def get_patch(image, center_x, center_y, dim=DEFAULT_PATCH_SIZE):
	
	return image[center_y-dim//2:center_y+dim//2 + 1,center_x-dim//2:center_x+dim//2 + 1]

def get_area(summed_table, i, j, dim=DEFAULT_PATCH_SIZE):
	
	if (i>dim//2) and (j >dim//2):
		return summed_table[i+dim//2,j+dim//2] + summed_table[i-dim//2-1,j-dim//2-1] - summed_table[i-dim//2-1,j+dim//2] - summed_table[i+dim//2, j - dim//2 - 1]
	elif i>dim//2:
		return summed_table[i+dim//2,j+dim//2] - summed_table[i-dim//2-1, j+dim//2]
	elif j>dim//2:
		return summed_table[i+dim//2,j+dim//2] - summed_table[i+dim//2, j-dim//2-1]
	else:
		return summed_table[i+dim//2,j+dim//2]

def scale_and_show(image):
	scale_coeff = 255/image[np.unravel_index(image.argmax(),image.shape)]
	# plt.imshow(image*scale_coeff,'gray')
	# plt.show()
	cv2.imshow('d', image*scale_coeff)

while(True):
	image = cv2.cvtColor(cap1.read()[1], cv2.COLOR_BGR2GRAY)
	image = cv2.flip(image, 0)
	# image = np.resize(image, (image.shape[0]//SCALE, image.shape[1]//SCALE))
	image = cv2.resize(image, dsize=(image.shape[1]//SCALE, image.shape[0]//SCALE))
	left_image = np.frombuffer(image.tobytes(), dtype=np.uint8).astype(np.int64)
	left_image = left_image.reshape((image.shape[0], image.shape[1]))
	image = cv2.cvtColor(cap2.read()[1], cv2.COLOR_BGR2GRAY)
	image = cv2.flip(image,0)
	image = cv2.resize(image, dsize=(image.shape[1]//SCALE, image.shape[0]//SCALE))
	right_image = np.frombuffer(image.tobytes(), dtype=np.uint8).astype(np.int64)
	right_image = right_image.reshape((image.shape[0], image.shape[1]))
	m,n = left_image.shape[0], left_image.shape[1]
	D_MAX = n//8

	p_max = np.full((m, n), -1, dtype=np.float)
	Dmap = np.zeros((m,n))
	p = np.zeros((m,n))
	p_avg = np.zeros((m,n))

	left_mu = np.zeros((m,n))
	right_mu = np.zeros((m,n))
	left_sigma = np.zeros((m,n))
	right_sigma = np.zeros((m,n))

	left_integral, left_squared_integral = cv2.integral2(left_image.astype(np.float))
	left_integral = left_integral[1:,1:]
	left_squared_integral = left_squared_integral[1:,1:]
	right_integral, right_squared_integral = cv2.integral2(right_image.astype(np.float))
	right_integral = right_integral[1:,1:]
	right_squared_integral = right_squared_integral[1:,1:]

	for i in range(DEFAULT_PATCH_SIZE//2, m-DEFAULT_PATCH_SIZE//2):
			for j in range(DEFAULT_PATCH_SIZE//2, n-DEFAULT_PATCH_SIZE//2):
				
				left_mu[i,j] = get_area(left_integral, i, j)/(DEFAULT_PATCH_SIZE**2)
				right_mu[i,j] = get_area(right_integral, i, j)/(DEFAULT_PATCH_SIZE**2)
				
				left_sigma[i,j] = np.sqrt(get_area(left_squared_integral, i, j)/(DEFAULT_PATCH_SIZE**2) - left_mu[i,j]**2)
				right_sigma[i,j] = np.sqrt(get_area(right_squared_integral, i, j)/(DEFAULT_PATCH_SIZE**2) - right_mu[i,j]**2)
	
	Image.fromarray(left_mu).show()


	for d in range(0, D_MAX):
		
		for center_x in range(DEFAULT_PATCH_SIZE//2,n-DEFAULT_PATCH_SIZE//2):
			for center_y in range(DEFAULT_PATCH_SIZE//2,m-DEFAULT_PATCH_SIZE//2):
				
				if center_x + d >= n - DEFAULT_PATCH_SIZE//2:
					continue
				if left_sigma[center_y, center_x]*right_sigma[center_y, center_x + d]==0:
					p[center_y, center_x] = 0
				else:
					p[center_y, center_x] = (np.mean(np.multiply(get_patch(left_image, center_x, center_y), get_patch(right_image, center_x + d, center_y))) \
											- left_mu[center_y,center_x]*right_mu[center_y, center_x + d]) \
											/ (left_sigma[center_y,center_x]*right_sigma[center_y, center_x + d])
		

		p_summed_table = p.cumsum(axis=0).cumsum(axis=1)
		for i in range(AVG_FILTER_SIZE//2, m-AVG_FILTER_SIZE//2):
			for j in range(AVG_FILTER_SIZE//2, n-AVG_FILTER_SIZE//2):
				p_avg[i,j] = get_area(p_summed_table, i, j, AVG_FILTER_SIZE)/AVG_FILTER_SIZE**2
				if p_avg[i,j] > p_max[i,j]:
					p_max[i,j] = p_avg[i,j]
					Dmap[i,j] = d

	scale_and_show(Dmap.astype(np.int64))