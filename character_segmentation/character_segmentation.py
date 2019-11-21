import cv2
img = cv2.imread('test.png', 0)
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
