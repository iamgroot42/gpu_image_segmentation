#include <algorithm>
#include <iostream>
#include <list>
#include <limits.h>
#include <queue>

#define FLATIMAGESIZE 316 * 300
#define N 256
#define M 256

using namespace std;

struct Edge
{
	int v;
	unsigned long long int capacity;
	Edge(int v = -1, unsigned long long int capacity = LLONG_MAX)
	{
		this -> v = v;
		this -> capacity = capacity;
	}
};

class Graph
{
public:
	unsigned long long int sourceEdges[N * M], sinkEdges[N * M];
	Edge edges[N * M * 10];

	list<int> adj[N * M];
	bool active[N * M], reachable[N * M];
	int labelCount[N * M], label[N * M];
	unsigned long long int excess[N * M];
	queue<int> activeVertices;

	int source, sink, V;

	Graph(int V);
	void setTerminals(int source, int sink);
	void addEdge(int u, int v, unsigned long long int capacity);
	void initializePreflow();
};

Graph::Graph(int V)
{
	cout << "CHAKKAYYYYYYYYYYYYYYYY\n";
	this -> V = V;
	cout << "HERE\n";
	// label = new int[V];
	// labelCount = new int[2 * V];
	// excess = new unsigned long long int[V];
	// active = new bool[V];
	// reachable = new bool[V];
	// adj = new list<int>[V];

	// sourceEdges = new unsigned long long int[N * M];
	// sinkEdges = new unsigned long long int[N * M];
	// edges = new Edge[N * M * 10];

	for (int i = 0; i < N * M; i++)
	{
		sourceEdges[i] = LLONG_MAX;
		sinkEdges[i] = LLONG_MAX;
	}
}

void Graph::setTerminals(int source, int sink)
{
	this -> source = source;
	this -> sink = sink;
}

void Graph::addEdge(int u, int v, unsigned long long int capacity)
{
	adj[u].push_back(v);

	if (u == this -> source)
	{
		if (this -> sourceEdges[v] == LLONG_MAX)
			this -> sourceEdges[v] = capacity;
		else
			this -> sourceEdges[v] += capacity;
	}
	else if (u == this -> sink)
	{
		if (this -> sinkEdges[v] == LLONG_MAX)
			this -> sinkEdges[v] = capacity;
		else
			this -> sinkEdges[v] += capacity;
	}
	else
	{
		bool flag = false;
		int pos = -1;
		for (int i = 0; i < 10; i++)
		{
			if (this -> edges[u * M + i].v == v)
			{
				flag = true;
				this -> edges[u * M + i].capacity += capacity;
			}
			else if (this -> edges[u * M + i].capacity == LLONG_MAX)
				pos = i;
		}

		if (!flag)
		{
			this -> edges[u * M + pos].v = v;
			this -> edges[u * M + pos].capacity = capacity;
		}
	}
}

void Graph::initializePreflow()
{
	for (int i = 0; i < this -> V; i++)
	{
		active[i] = false;
		excess[i] = 0;
		label[i] = 0;
		labelCount[2 * i] = 0;
		labelCount[2 * i + 1] = 0;
	}
	label[this -> source] = this -> V;
	labelCount[0] = this -> V - 1;
	labelCount[this -> V] = 1;
	for (int i = 0; i < this -> V; i++)
		if (this -> sourceEdges[i] < LLONG_MAX)
			excess[this -> source] += this -> sourceEdges[i];
}

__global__ void pushKernel(Graph *g)
{
	// __shared__ unsigned long long int label[M * N];
	// __shared__ unsigned long long int excess[M * N];

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int u = i * M + j;

	// if (i < N && j < M)
	// {
	// 	label[u] = globalLabels[u];
	// 	excess[u] = globalExcesses[u];
	// }
	// __syncthreads();

	unsigned long long int label = g -> label[u], excess = g -> excess[u], capacity = 0, diff = 0;
	if (u == g -> source)
		for (int v = 0; v < N * M; v++)
			if (g -> label[v] < label)
			{
				capacity = g -> sourceEdges[v];
				diff = min(excess, capacity);
				capacity -= diff;
				if (v == g -> sink)
					atomicAdd(&g -> sinkEdges[u], diff);
				else
					for (int l = 0; l < 10; l++)
						if (g -> edges[v * 10 + l].v == u)
							atomicAdd(&g -> edges[v * 10 + l].capacity, diff);
				g -> sourceEdges[v] = capacity;
			}
	else if (u == g -> sink)
		for (int v = 0; v < N * M; v++)
			if (g -> label[v] < label)
			{
				capacity = g -> sinkEdges[v];
				diff = min(excess, capacity);
				capacity -= diff;
				if (v == g -> source)
					atomicAdd(&g -> sourceEdges[u], diff);
				else
					for (int l = 0; l < 10; l++)
						if (g -> edges[v * 10 + l].v == u)
							atomicAdd(&g -> edges[v * 10 + l].capacity, diff);
				g -> sinkEdges[v] = capacity;
			}
	else
		for (int l = 0; l < 10; l++)
		{
			int v = g -> edges[u * 10 + l].v;
			if (g -> label[v] < label)
			{
				capacity = g -> edges[u * 10 + l].capacity;
				diff = min(excess, capacity);
				capacity -= diff;
				if (v == g -> source)
					atomicAdd(&g -> sourceEdges[u], diff);
				if (v == g -> sink)
					atomicAdd(&g -> sinkEdges[u], diff);
				else
					for (int m = 0; m < 10; m++)
						if (g -> edges[v * 10 + m].v == u)
							atomicAdd(&g -> edges[v * 10 + m].capacity, diff);
				g -> edges[u * 10 + l].capacity = capacity;
			}
		}
}

__global__ void localRelabel(Graph *g)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int u = i * M + j, v, l;

	bool label = g -> label[u];
	unsigned long long int minLabel = LLONG_MAX;

	for (l = 0; l < 10; l++)
		if (g -> edges[u * 10 + l].capacity != LLONG_MAX)
		{
			v = g -> edges[u * 10 + l].v;
			if (minLabel > g -> label[v])
			{
				minLabel = g -> label[v];
				label = minLabel + 1;
			}
		}
	g -> label[u] = label;
}

__global__ void globalRelabel(Graph *g, int k)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int u = i * M + j, l;
	if (k == 1)
	{
		if (g -> sinkEdges[u] > 0)
			g -> label[u] = 1;
		__syncthreads();
		return;
	}

	unsigned long long int label = g -> label[u];
	if (label == LLONG_MAX)
	{
		if (u == g -> source || u == g -> sink)
			for (l = 0; l < M * N; l++)
				if (g -> label[l] == k)
				{
					label = k + 1;
					break;
				}
		else
		{
			for (l = 0; l < 10; l++)
				if (g -> label[u * 10 + l] == k)
				{
					label = k + 1;
					break;
				}
		}
	}
	g -> label[u] = label;
	__syncthreads();
}

int main()
{
	int n, m, x, y;
	unsigned long long int z;
	Graph *g = new Graph(N * M);
	g -> setTerminals(0, N * M - 1);
	while (m--)
	{
		cin >> x >> y >> z;
		if (x != y && z > 0)
			g -> addEdge(x - 1, y - 1, z);
	}
	g -> initializePreflow();

	Graph *g_gpu;
	cudaMalloc((void**)&g_gpu, sizeof(Graph));
	cudaMemcpy(g_gpu, &g, sizeof(Graph), cudaMemcpyHostToDevice);
	dim3 numBlocks(16, 16);
	dim3 threadsPerBlock(M, N);

	pushKernel<<<numBlocks, threadsPerBlock>>>(g_gpu);
}