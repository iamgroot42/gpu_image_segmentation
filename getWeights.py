import numpy as np
from PIL import Image
import sys

LAMBDA = 2
SIGMA = 1 
B_PROPORTIONALITY = 1
N_BINS = 10
BINS = int(255.0 / N_BINS)
EPSILON = 0.001


class Graph:

	def __init__(self, V):
		self.adjlist = {}
		self.constraints = [] # -1: none, 0: object, 1: background
		self.V = V
		for i in range(V):
			self.constraints.append(-1)
			self.adjlist[i] = {}

	def addEdge(self, u, v, cap):
		if v in self.adjlist[u]:
			self.adjlist[u][v] += cap
		else:
			self.adjlist[u][v] = cap
		if u in self.adjlist[v]:
			self.adjlist[v][u] += cap
		else:
			self.adjlist[v][u] = cap

def background_function(pixmap, u, v):
	u_x, u_y = u
	v_x, v_y = v
	norm = np.linalg.norm(pixmap[u_x][u_y] - pixmap[v_x][v_y]) ** 2
	distance = np.linalg.norm(np.array(u) - np.array(v))
	cost = B_PROPORTIONALITY * np.exp( -norm / (2 * SIGMA * SIGMA) ) / distance
	return cost


def region_function(pixmap, u, region, histo):
	(i, j) = u
	r, g, b = pixmap[i][j]
	prob = - np.log( histo[r/BINS][g/BINS][b/BINS] + EPSILON)
	return prob


def createWeights(pixmap, G, height, width, OH, BH):
	# V = P U {S(object), T(background)}
	n_nodes = (height * width) + 2
	K = 0
	# {p,q} type edges
	for i in range(width):
		for j in range(height):
			internal_sum = 0 
			for k in range(-1, 2):
				for l in range(-1, 2):
					if (k != 0 or l != 0):
						u = i * height + j
						v = (i + k) * height + (j + l)
						if ( ( i + k >= 0 ) and (i + k < width) and (j + l >= 0) and (j + l < height) ):
							B_uv = background_function(pixmap, (i, j), (i + k, j + l))
							internal_sum += B_uv
							G.addEdge(u + 1, v + 1, B_uv)
			K = max(internal_sum, K)
	K += 1
	# {p,S} type edges
	for i in range(width):
		for j in range(height):
			u = (i * height) + j
			if (G.constraints[u] == 0):
				G.addEdge(u + 1, 0, K)
			# elif (G.constraints[u] == -1):
			# 	weight = LAMBDA * region_function(pixmap, (i,j), 'bkg', BH)
			# 	G.addEdge(u + 1, 0, weight)
	# {p,T} type edges
	v = n_nodes - 1
	for i in range(width):
		for j in range(height):
			u = (i * height) + j
			if (G.constraints[u] == 1):
				G.addEdge(u + 1, v, K)
			# elif (G.constraints[u] == -1):
			# 	weight = LAMBDA * region_function(pixmap, (i,j), 'obj', OH)
			# 	G.addEdge(u + 1, v, weight)
	return G

def load_constraints(image, filename, G, constraint_type, height):
	f = open(filename, 'r')

	freqs = []

	for line in f:
		x, y = line.rstrip('\n').split(' ')
		x, y = int(x), int(y)
		co_or = y * height + x
		G.constraints[co_or] = constraint_type
		freqs.append([ image[y][x][0], image[y][x][1], image[y][x][2] ])

	freqs = np.array(freqs)
	H = np.histogramdd(freqs, bins=N_BINS, normed=True, range=((0,255), (0,255), (0,255)))[0]
	return H


def main(image_name, object_name, background_name):
	image = np.array(Image.open(image_name).convert('RGB'))
	width, height = image.shape[:2]

	n_nodes = (width * height) + 2

	image = np.array(image)

	G = Graph(n_nodes)

	OH = load_constraints(image, object_name, G, 0, height)
	BH = load_constraints(image, background_name, G, 1, height)
	
	createWeights(image, G, height, width, OH, BH)

	E = 0
	for i in range(G.V):
		E += len(G.adjlist[i].keys())
	print G.V, E

	for i in range(G.V):
		for j in G.adjlist[i].keys():
			if (G.adjlist[i][j] > 0):
				print i + 1, j + 1, G.adjlist[i][j]
	

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2], sys.argv[3])
