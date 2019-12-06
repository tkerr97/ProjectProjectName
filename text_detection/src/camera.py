import cv2
import os

# Camera class used to take picture of document
class Camera:
	def __init__(self, camera_number, img_name):
		self.camera_number = camera_number
		self.img_name = img_name

	def capture(self):
		captured_img = False
		cam = cv2.VideoCapture(self.camera_number)
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
				img_dir = os.path.join('../data', self.img_name)
				cv2.imwrite(img_dir, frame)
				print("{} written!".format(self.img_name))
				captured_img = True
				break

		cam.release()
		cv2.destroyAllWindows()
		
		return captured_img