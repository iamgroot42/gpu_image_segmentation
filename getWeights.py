import numpy as np
from PIL import Image
import sys

LAMBDA = 2
SIGMA = 1 
B_PROPORTIONALITY = 1


class Graph:

	def __init__(self, V):
		self.adjlist = {}
		self.V = V
		for i in range(V):
			self.adjlist[i] = {}

	def addEdge(self, u, v, cap):
		if v in self.adjlist[u].keys():
			self.adjlist[u][v] += cap
		else:
			self.adjlist[u][v] = cap
		if u in self.adjlist[v].keys():
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


def createWeights(pixmap, height, width, hard_object, hard_background):
	# V = P U {S(object),T(background)}
	n_nodes = (height*width) + 2;
	G = Graph(n_nodes)
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
			if (u,0) in hard_object:
				g,addEdge(u, 0, K)
				g,addEdge(0, u, K)
			elif (u,0) not in hard_background:
				G.addEdge(u, 0, LAMBDA * region_function(pixmap, u, 0, 'bkg'))
				G.addEdge(0, u, LAMBDA * region_function(pixmap, u, 0, 'bkg'))
	# {p,T} type edges		
	for i in range(height):
		for j in range(width):
			u = (i * width) + j
			v = n_nodes - 1
			if (u,v) in hard_background:
				G.addEdge(u, v, K)
				G.addEdge(v, u, K)
			elif (u,v) not in hard_object:
				G.addEdge(u , v, LAMBDA * region_function(pixmap, u, v, 'obj'))
				G.addEdge(v, u , LAMBDA * region_function(pixmap, u, v, 'obj'))
	return G


def load_constraints(filename):
	f = open(filename, 'r')
	points = set()

	for line in f:
		x, y = line.rstrip('\n').split(' ')
		points.add((x,y))
	return points


def main(image_name, object_name, background_name):
	image = Image.open(image_name)
	width, height = image.size

	image = np.array(image)

	hard_object = load_constraints(object_name)
	hard_background = load_constraints(background_name)
	
	G = createWeights(image, height, width, hard_object, hard_background)

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
