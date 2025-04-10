
import cv2

def read_resize_blur(img_path, roi_size):
	background_image = cv2.imread(img_path)
	background_image = cv2.cvtColor(background_image,cv2.COLOR_BGR2RGB)
	background_image = cv2.resize(background_image, roi_size, interpolation=cv2.INTER_AREA)
	background_image = cv2.blur(background_image, (20,20))
	return background_image



