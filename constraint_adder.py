import cv2
import sys

object_points = []
background_points = []
counter = 0
data = None

def mouse_callback(event, x, y, flags, params):
	global object_points
	global background_points
	global counter
	global data
	if event == cv2.EVENT_LBUTTONDOWN:
		object_points.append((x, y))
	elif event == cv2.EVENT_RBUTTONDOWN:
		background_points.append((x, y))


def annotate_images(img_path):
	global data
	data = cv2.imread(img_path)
	cv2.imshow('Image',data)
	cv2.setMouseCallback('Image', mouse_callback)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def write_points(data, filename):
	f = open(filename, 'w')
	for point in data:
		x,y = point
		f.write(str(x) + " " + str(y) +  "\n")
	f.close()


if __name__ == "__main__":
	file_path = sys.argv[1]
	print("Left click to label object points")
	print("Right click to label background points")
	annotate_images(file_path)
	write_points(object_points, "OBJECT")
	write_points(background_points, "BACKGROUND")
