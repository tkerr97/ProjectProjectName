Text detection model based on IAM dataset

Description:
	Run src/main.py to use the model. The program will access a connected 
	webcam and display its image. Hit the space key to take a picture of 
	a document. The taken picture will be segmented by word and will 
	print an array of all the recognized text. 

	The taken image can be found in the data/ directory and the segmented 
	word images are placed in the out/ directory.

	The pretrained text detection model used can be found at https://github.com/githubharald/SimpleHTR
	The word segmentation model is based on code found at https://github.com/githubharald/WordSegmentation