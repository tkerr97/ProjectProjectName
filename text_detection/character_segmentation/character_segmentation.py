import cv2
import os
import numpy as np
import shutil

from WordSegmentation import wordSegmentation, prepareImg

def word_segment(img_name):
	# separate document image into word images
	print('Segmenting words of sample %s'%img_name)

	# read image, prepare it by resizing it to fixed height and converting it to grayscale
	img = prepareImg(cv2.imread(img_name), 50)

	# execute segmentation with given parameters
	# -kernelSize: size of filter kernel (odd integer)
	# -sigma: standard deviation of Gaussian function used for filter kernel
	# -theta: approximated width/height ratio of words, filter function is distorted by this factor
	# - minArea: ignore word candidates smaller than specified area
	res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)

	# write output to 'out/inputFileName' directory
	if os.path.exists('./%s'%img_name):
		shutil.rmtree('./%s'%img_name)
		os.mkdir('./%s'%img_name)
	else:
		os.mkdir('./%s'%img_name)

	# iterate over all segmented words
	print('Segmented into %d words'%len(res))
	for (j, w) in enumerate(res):
		(wordBox, wordImg) = w
		(x, y, w, h) = wordBox
		cv2.imwrite('./%s/%d/%d.png'%(img_name, j), wordImg, wordImg) # save word
		cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image

	# output summary image with bounding boxes around words
	cv2.imwrite('./%s/summary.png'%img_name, img)

def segment(file_name: str):
    img = cv2.imread(file_name, 0)
    cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,img)
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    cv2.imshow("contours", img)
    cv2.waitKey(0)
    d=0
    for ctr in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi = img[y:y+h, x:x+w]

        cv2.imshow('character: %d'%d,roi)
        cv2.imwrite('character_%d.png'%d, roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        d+=1

def capture(camera_number: int , img_name: str):
    cam = cv2.VideoCapture(camera_number)
    cv2.namedWindow('Text Detection')

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            captured_img = False
            break
        elif k%256 == 32:
            # SPACE pressed
            img_dir = img_name
            cv2.imwrite(img_dir, frame)
            print("{} written!".format(img_name))
            captured_img = True
            break

def increase_contrast(img_name: str):
	# load image
	img_dir = img_name
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

	# img_gray = 255 - img_gray #invert color back

    # increase line width
	kernel = np.ones((3, 3), np.uint8)
	processed_img = cv2.erode(img_gray, kernel, iterations = 1)

	# save processed image
	img_name_root = os.path.splitext(img_name)[0]
	processed_img_name = img_name_root + '_processed.png'
	processed_img_dir = processed_img_name
	cv2.imwrite(processed_img_dir, processed_img)

	return processed_img_name

def main():
    capture(1, 'test_img.png')
    increase_contrast('test_img.png')
    word_segment('test_img_processed.png')
    # segment('test_img_processed.png')

if __name__ == '__main__':
    main()
