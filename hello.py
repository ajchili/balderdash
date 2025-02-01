import cv2


def resize(img, desired_width = 600):
	(h, w) = img.shape[:2]
	aspect_ratio = h / w
	new_height = int(desired_width * aspect_ratio)
	return cv2.resize(img, (desired_width, new_height))


for image_path in ["./images/IMG_4705.png", "./images/IMG_5411.png", "./images/IMG_8512.JPEG"]:
	img = cv2.imread(image_path)
	img = resize(img)
	cv2.imshow("Image", img)
	cv2.waitKey(0)


cv2.destroyAllWindows()