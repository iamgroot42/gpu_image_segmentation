import sys
import cv2
import numpy as np
from PIL import Image



def mask_image(image, coordinates):
	mask = np.zeros(image.shape[:2])
	mask = mask.flatten()
	for (x, y) in coordinates:
		mask[x] = 255
		mask[y] = 255
	return mask.reshape(image.shape[:2])


def load_points(filename):
	f = open(filename, 'r')
	points = []
	for row in f:
		x,y = row.rstrip('\n').split(' ')
		x,y = int(x), int(y)
		points.append((x,y))
	return points


if __name__ == "__main__":
	image = Image.open(sys.argv[1])
	image = cv2.imread(sys.argv[1])
	points = load_points(sys.argv[2])
	mask = mask_image(image, points)
	# mask = mask.astype('uint8')
	cv2.imwrite('images/segmented.png', mask)
	# print mask
	# im = Image.fromarray(mask)
	# im.save("images/segmented.png")
