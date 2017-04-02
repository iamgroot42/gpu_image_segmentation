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
	u_x , u_y = u
	v_x, v_y = v
	norm = np.linalg.norm(pixmap[u_x][u_y] - pixmap[v_x][v_y]) ** 2
	distance = np.linalg.norm(np.array(u) - np.array(v))
	cost = B_PROPORTIONALITY * np.exp( - norm / (2 * SIGMA * SIGMA)) / distance
	return int(cost)


def region_function(pixmap, u, v, region):
	return 1 * (region == 'bkg')


def createWeights(pixmap, G, height, width):
	# V = P U {S(object),T(background)}
	n_nodes = (height*width) + 2
	K = 0
	# {p,q} type edges
	for i in range(1, height-1, 2):
		for j in range(1, width-1, 2):
			internal_sum = 0 
			for k in range(-1,2):
				for l in range(-1,2):
					if k or l:
						B_uv = background_function(pixmap, (i,j), (i+k, j+l))
						internal_sum += B_uv
						u = i * width + j
						v = (i + k) * width + (j + l)
						G.addEdge(u, v , B_uv)
						G.addEdge(v, u , B_uv)
			K = max(internal_sum ,K)
	K += 1
	# {p,S} type edges
	for i in range(height):
		for j in range(width):
			u = (i * width) + j
			if G.constraints[u] == 0:
				G.addEdge(u, 0, K)
				G.addEdge(0, u, K)
			elif G.constraints[u] == -1:
				weight = LAMBDA * region_function(pixmap, u, 0, 'bkg')
				G.addEdge(u, 0, 3)
				G.addEdge(0, u, 3)
	# {p,T} type edges		
	v = n_nodes - 1
	for i in range(height):
		for j in range(width):
			u = (i * width) + j
			if G.constraints[u] == 1:
				G.addEdge(u, v, K)
				G.addEdge(v, u, K)
			elif G.constraints[u] == -1:
				weight = LAMBDA * region_function(pixmap, u, v, 'obj')
				G.addEdge(u, v, weight)
				G.addEdge(v, u, weight)
	return G


def load_constraints(filename, G, constraint_type, width):
	f = open(filename, 'r')

	for line in f:
		x, y = line.rstrip('\n').split(' ')
		G.constraints[int(x) * width + int(y)] = constraint_type


def main(image_name, object_name, background_name):
	image = Image.open(image_name)
	width, height = image.size
	n_nodes = (height*width) + 2

	image = np.array(image)

	G = Graph(n_nodes)

	load_constraints(object_name, G, 0, width)
	load_constraints(background_name, G, 1, width)
	
	createWeights(image, G, height, width)

	E = 0
	for i in range(G.V):
		E += len(G.adjlist[i].keys())

	print G.V, E

	for i in range(G.V):
		for j in G.adjlist[i].keys():
			if G.adjlist[i][j] > 0:
				print i + 1, j + 1, G.adjlist[i][j]
	

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2], sys.argv[3])
