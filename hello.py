import cv2
import numpy as np

HOLDS_HUE_RANGES = [
	# Orange
	[0, 45],
	# Yellow
	[45, 65],
	# Green
	[65, 140],
	# Blue
	[210, 250],
	# Purple
	[250, 270],
	# Pink
	[290, 320]
]

def resize(img, desired_width = 600):
	(h, w) = img.shape[:2]
	aspect_ratio = h / w
	new_height = int(desired_width * aspect_ratio)
	return cv2.resize(img, (desired_width, new_height))

def quantize(img, k = 8):
	pixels = img.reshape((-1, 3))
	pixels = np.float32(pixels)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

	_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	centers = np.uint8(centers)

	quantized_image = centers[labels.flatten()]
	quantized_image = quantized_image.reshape(img.shape)

	return quantized_image

def hsl_to_hsv(hue, saturation, luminance):
	return np.array([hue / 2, (saturation / 100) * 255, (luminance / 100) * 255])


for image_path in ["./images/IMG_4705.png", "./images/IMG_5411.png", "./images/IMG_8512.JPEG"]:
	print(f"Loading image {image_path}")
	img = cv2.imread(image_path)
	img = resize(img)
	img = quantize(img, k = 100)
	cv2.imshow("Image", img)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	cv2.imshow("HSV", hsv)

	cv2.moveWindow("Image", 0, 0)
	cv2.moveWindow("HSV", img.shape[1] + 20, 0)

	for ranges in HOLDS_HUE_RANGES:
		[low, high] = ranges
		print(f"quantizing for {ranges=}")
		mask = cv2.inRange(hsv, hsl_to_hsv(*[low, 50, 50]), hsl_to_hsv(*[high, 100, 100]))
		result = cv2.bitwise_and(img, img, mask=mask)
		result = quantize(result, k = 2)
		cv2.imshow("Mask Result", result)
		cv2.moveWindow("Mask Result", (img.shape[1] + 20) * 2, 0)

		cv2.waitKey(0)


cv2.destroyAllWindows()