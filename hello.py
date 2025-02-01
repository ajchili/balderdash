import cv2
import numpy as np


YELLOW_HOLD_HSL_LIGHT = [45, 50, 50]
YELLOW_HOLD_HSL_DARK = [65, 100, 100]

def resize(img, desired_width = 600):
	(h, w) = img.shape[:2]
	aspect_ratio = h / w
	new_height = int(desired_width * aspect_ratio)
	return cv2.resize(img, (desired_width, new_height))

def hsl_to_hsv(hue, saturation, luminance):
	return np.array([hue / 2, (saturation / 100) * 255, (luminance / 100) * 255])


for image_path in ["./images/IMG_4705.png", "./images/IMG_5411.png", "./images/IMG_8512.JPEG"]:
	img = cv2.imread(image_path)
	img = resize(img)
	cv2.imshow("Image", img)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	cv2.imshow("HSV", hsv)
	mask = cv2.inRange(hsv, hsl_to_hsv(*YELLOW_HOLD_HSL_LIGHT), hsl_to_hsv(*YELLOW_HOLD_HSL_DARK))
	result = cv2.bitwise_and(img, img, mask=mask)
	cv2.imshow("Mask Result", result)

	cv2.moveWindow("Image", 0, 0)
	cv2.moveWindow("HSV", img.shape[1] + 20, 0)
	cv2.moveWindow("Mask Result", (img.shape[1] + 20) * 2, 0)

	cv2.waitKey(0)


cv2.destroyAllWindows()