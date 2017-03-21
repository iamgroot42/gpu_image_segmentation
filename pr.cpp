#include <assert.h>
#include <algorithm>
#include <iostream>
#include <queue>
#include <utility>

using namespace std;

struct Edge
{
	long long capacity, flow;
	Edge(long long capacity = 0, long long flow = -1)
	{
		this -> capacity = capacity;
		this -> flow = flow;
	}
};

pair <Edge, Edge> edges[5010][5010];

class Graph
{
public:
	int V;
	long long *label, *excessFlow;
	bool *active;
	queue<int> activeNodes;

	Graph(int V);
	void addEdge(int u, int v, 	long long capacity, long long flow, int color);
	void initializePreflow(int source, int sink);
};

Graph::Graph(int V)
{
	this -> V = V;
	label = new long long[V];
	excessFlow = new long long[V];
	active = new bool[V];
}

void Graph::addEdge(int u, int v, long long capacity, long long flow = 0, int color = 0)
{
	edges[u][v].first.capacity += capacity;
	edges[u][v].first.flow += flow;

	edges[v][u].second.capacity = 0;
	edges[v][u].second.flow = 0;
}

void Graph::initializePreflow(int source, int sink)
{
	for (int i = 0; i < this -> V; i++)
	{
		label[i] = 0;
		excessFlow[i] = 0;
		active[i] = false;
	}
	label[source] = this -> V;
	for (int i = 0; i < this -> V; i++)
		excessFlow[source] += edges[source][i].first.capacity;
}

int main()
{
	int i, m, n, x, y, z;
	cin >> n >> m;
	Graph g(n);
	while (m--)
	{
		cin >> x >> y >> z;
		g.addEdge(x - 1, y - 1, z);
	}
	return 0;
}