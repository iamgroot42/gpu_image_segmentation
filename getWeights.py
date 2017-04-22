import numpy as np
from PIL import Image
import sys

LAMBDA = 2
SIGMA = 1 
B_PROPORTIONALITY = 1


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
	return int(cost) + 1

def region_function(pixmap, u, region):
	return 1 + (region == 'bkg')

def createWeights(pixmap, G, height, width):
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
							# G.addEdge(u + 1, v + 1, 1)
			K = max(internal_sum, K)
	K += 1
	# {p,S} type edges
	for i in range(width):
		for j in range(height):
			u = (i * height) + j
			if (G.constraints[u] == 0):
				G.addEdge(u + 1, 0, K)
				# G.addEdge(u + 1, 0, 5)
			# elif (G.constraints[u] == -1):
			# 	weight = LAMBDA * region_function(pixmap, u, 'bkg')
			# 	G.addEdge(u + 1, 0, weight)
				# G.addEdge(u + 1, 0, 7)
	# {p,T} type edges
	v = n_nodes - 1
	for i in range(width):
		for j in range(height):
			u = (i * height) + j
			if (G.constraints[u] == 1):
				G.addEdge(u + 1, v, K)
				# G.addEdge(u + 1, v, 5)
			# elif (G.constraints[u] == -1):
			# 	weight = LAMBDA * region_function(pixmap, u, 'obj')
			# 	G.addEdge(u + 1, v, weight)
				# G.addEdge(u + 1, v, 7)
	return G

def load_constraints(filename, G, constraint_type, height):
	f = open(filename, 'r')

	for line in f:
		x, y = line.rstrip('\n').split(' ')
		G.constraints[int(y) * height + int(x)] = constraint_type


def main(image_name, object_name, background_name):
	image = np.array(Image.open(image_name))
	width, height = image.shape[:2]

	n_nodes = (width * height) + 2

	image = np.array(image)

	G = Graph(n_nodes)

	load_constraints(object_name, G, 0, height)
	load_constraints(background_name, G, 1, height)
	
	createWeights(image, G, height, width)

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
