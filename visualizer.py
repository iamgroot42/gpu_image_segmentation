import sys
import cv2
import numpy as np
from PIL import Image



def mask_image(image, coordinates):
	mask = np.zeros(image.shape[:2])
	mask = mask.flatten()
	print mask.shape
	# for (x, y) in coordinates:
	for x in coordinates:
		try:
			mask[x - 1] = 255
		except:
			print len(mask), x
		# mask[y] = 255
	return mask.reshape(image.shape[:2])


def load_points(filename):
	f = open(filename, 'r')
	points = []
	for row in f:
		x,y = row.rstrip('\n').split(' ')
		x,y = int(x), int(y)
		points.append((x,y))
	return points

def load_object_points(filename):
	f = open(filename, 'r')
	points = []
	for row in f:
		# x = row.rstrip('\n').split(' ')
		# x,y = int(x), int(y)
		# points.append((x,y))
		x = int(row.rstrip('\n'))
		points.append(x)
	return points


if __name__ == "__main__":
	# image = Image.open(sys.argv[1])
	image = cv2.imread(sys.argv[1])
	points = load_object_points(sys.argv[2])
	mask = mask_image(image, points)
	# mask = mask.astype('uint8')
	cv2.imwrite('images/segmented.png', mask)
	# print mask
	# im = Image.fromarray(mask)
	# im.save("images/segmented.png")
