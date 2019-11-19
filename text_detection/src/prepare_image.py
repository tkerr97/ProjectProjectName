import numpy as np
import cv2
import os


def increase_contrast(img_name: str):
	# load image
	img_dir = os.path.join('../data/', img_name)
	img_gray = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)

	# increase contrast
	threshold = 150
	img_gray = 255 - img_gray  # invert color
	for row in range(img_gray.shape[0]):
		for col in range(img_gray.shape[1]):
			# print(img_gray[row][col])
			if (img_gray[row][col] < threshold):
				img_gray[row][col] = 0
			else:
				img_gray[row][col] = 255

	img_gray = 255 - img_gray #invert color back

    # increase line width
	kernel = np.ones((3, 3), np.uint8)
	processed_img = cv2.erode(img_gray, kernel, iterations = 1)

	# save processed image
	img_name_root = os.path.splitext(img_name)[0]
	processed_img_name = img_name_root + '_processed.png'
	processed_img_dir = os.path.join('../data/', processed_img_name)
	cv2.imwrite(processed_img_dir, processed_img)

	return processed_img_name
