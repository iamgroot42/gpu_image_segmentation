import sys
import numpy as np
from PIL import Image


def mask_image(image, coordinates):
	image = np.array(image)
	flat_img = image.flatten()
	mask = np.ones(flat_img.shape)
	for (x,y) in coordinates:
		if x == 1:
			mask[y] = 0
		elif y == 1:
			mask[x]= 0
	masked = flat_img * mask
	masked = masked.reshape(image.shape)
	return masked


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
	points = load_points(sys.argv[2])
	mask = mask_image(image, points)
	mask = mask.astype('uint8')
	im = Image.fromarray(mask)
	im.save("segmented.jpeg")
