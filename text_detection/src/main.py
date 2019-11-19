from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess

from camera import Camera
from prepare_image import increase_contrast

from WordSegmentation import wordSegmentation, prepareImg

import os
import shutil


class FilePaths:
	"filenames and paths to data"
	fnCharList = '../model/charList.txt'
	fnAccuracy = '../model/accuracy.txt'
	fnTrain = '../data/'
	fnCorpus = '../data/corpus.txt'
	

def infer(model, fnImg):
	"recognize text in image provided by file path"
	img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
	batch = Batch(None, [img])
	(recognized, probability) = model.inferBatch(batch, True)
	print('Recognized:', '"' + recognized[0] + '"')
	print('Probability:', probability[0])
	return recognized


def main():
	
	# optional command line args
	parser = argparse.ArgumentParser()

	parser.add_argument('--image_name', type=str, default='test_img.png') # make sure this ends in an image file extension
	parser.add_argument('--camera_number', type=int, default=0)

	args = parser.parse_args()

	img_name = args.image_name
	camera_number = args.camera_number 

	decoderType = DecoderType.BestPath

	# capture image of document
	cam = Camera(camera_number, img_name)
	got_img = cam.capture()
	if not got_img:
		return

	# prepare image for word segmentation
	processed_img_name = increase_contrast(img_name)

	# separate document image into word images
	print('Segmenting words of sample %s'%img_name)
	
	# read image, prepare it by resizing it to fixed height and converting it to grayscale
	img = prepareImg(cv2.imread('../data/%s'%processed_img_name), 50)
	
	# execute segmentation with given parameters
	# -kernelSize: size of filter kernel (odd integer)
	# -sigma: standard deviation of Gaussian function used for filter kernel
	# -theta: approximated width/height ratio of words, filter function is distorted by this factor
	# - minArea: ignore word candidates smaller than specified area
	res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
	
	# write output to 'out/inputFileName' directory
	if os.path.exists('../out/%s'%img_name):
		shutil.rmtree('../out/%s'%img_name)
		os.mkdir('../out/%s'%img_name)
	
	# iterate over all segmented words
	print('Segmented into %d words'%len(res))
	for (j, w) in enumerate(res):
		(wordBox, wordImg) = w
		(x, y, w, h) = wordBox
		cv2.imwrite('../out/%s/%d.png'%(img_name, j), wordImg) # save word
		cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
	
	# output summary image with bounding boxes around words
	cv2.imwrite('../out/%s/summary.png'%img_name, img)

	
	# analyze words
	text = []
	print(open(FilePaths.fnAccuracy).read())
	model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=False)
	for word in os.listdir(f'../out/{img_name}'):
		if word != 'summary.png':
			new_word = infer(model, os.path.join('../out/', img_name, word))
			text.append(new_word)
			

	print(f'Document Text: {text}')
	


if __name__ == '__main__':
	main()

