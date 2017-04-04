#include <algorithm>
#include <iostream>
#include <list>
#include <limits.h>
#include <queue>

#define FLATIMAGESIZE 316 * 300

using namespace std;

struct Edge
{
	int v;
	long long capacity;
	Edge(int v = -1, long long capacity = -1)
	{
		this -> v = v;
		this -> capacity = capacity;
	}
};

long long sourceEdges[FLATIMAGESIZE + 10], sinkEdges[FLATIMAGESIZE + 10];
Edge edges[FLATIMAGESIZE + 10][10];

class Graph
{
public:
	list<int> *adj;
	bool *active, *reachable;
	int *labelCount, *label;
	long long *excess;
	queue<int> activeVertices;

	int source, sink, V;

	Graph(int V);
	void setTerminals(int source, int sink);
	void addEdge(int u, int v, long long capacity);
	void BFS();

	void initializePreflow();
	void markActive(int u);
	void relabel(int u);
	void push(int u, int v);
	void gap(int k);

	long long result();
};

Graph::Graph(int V)
{
	this -> V = V;
	label = new int[V];
	labelCount = new int[2 * V];
	excess = new long long[V];
	active = new bool[V];
	reachable = new bool[V];
	adj = new list<int>[V];
}

void Graph::setTerminals(int source, int sink)
{
	this -> source = source;
	this -> sink = sink;
}

void Graph::addEdge(int u, int v, long long capacity)
{
	adj[u].push_back(v);

	if (u == this -> source)
	{
		if (sourceEdges[v] < 0)
			sourceEdges[v] = capacity;
		else
			sourceEdges[v] += capacity;
	}
	else if (u == this -> sink)
	{
		if (sinkEdges[v] < 0)
			sinkEdges[v] = capacity;
		else
			sinkEdges[v] += capacity;
	}
	else
	{
		bool flag = false;
		int pos = -1;
		for (int i = 0; i < 10; i++)
		{
			if (edges[u][i].v == v)
			{
				flag = true;
				edges[u][i].capacity += capacity;
			}
			else if (edges[u][i].capacity == -1)
				pos = i;
		}

		if (!flag)
		{
			edges[u][pos].v = v;
			edges[u][pos].capacity = capacity;
		}
	}
}

// void Graph::BFS()
// {
// 	queue<int> neighbours;
// 	list<int>::iterator iter;
// 	reachable[this -> sink] = true;
// 	neighbours.push(this -> sink);

// 	while (!neighbours.empty())
// 	{
// 		int x = neighbours.front(), y;
// 		neighbours.pop();

// 		for (iter = adj[x].begin(); iter != adj[x].end(); iter++)
// 		{
// 			y = *iter;
// 			if (!reachable[y] && y != this -> source)
// 				if ( (x == sink && sinkEdges[y] > 0) || (x != source && x != sink && edges[x][y] > 0) )
// 				{
// 					reachable[y] = true;
// 					neighbours.push(y);
// 				}
// 		}
// 	}
// }

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
		if (sourceEdges[i] >= 0)
			excess[this -> source] += sourceEdges[i];
}

void Graph::markActive(int u)
{
	if (!active[u] && excess[u] > 0)
	{
		active[u] = true;
		activeVertices.push(u);
	}
}

void Graph::gap(int k)
{
	for (int i = 0; i < this -> V; i++)
	{
		if (label[i] < k)
			continue;
		labelCount[label[i]]--;
		label[i] = max(label[i], this -> V + 1);
		labelCount[label[i]]++;
		markActive(i);
	}
}

void Graph::relabel(int u)
{
	labelCount[label[u]]--;
	int i, minLabel = INT_MAX;

	// Source and sink are never relabeled.
	for (i = 0; i < 10; i++)
		if (edges[u][i].capacity > 0)
		{
			minLabel = min(minLabel, label[edges[u][i].v]);
			label[u] = minLabel + 1;
		}
	labelCount[label[u]]++;
	markActive(u);
}

void Graph::push(int u, int v)
{
	long long diff = excess[u];
	int i, pos = -1;

	if (u == this -> source && sourceEdges[v] > 0)
		diff = min(diff, sourceEdges[v]);

	else if (u == this -> sink && sinkEdges[v] > 0)
		diff = min(diff, sinkEdges[v]);

	else if (u != this -> source && u != this -> sink)
		for (i = 0; i < 10; i++)
			if (edges[u][i].v == v)
			{
				pos = i;
				diff = min(diff, edges[u][i].capacity);
			}

	if (diff == 0 || label[u] <= label[v])
		return;

	excess[u] -= diff;
	excess[v] += diff;

	if (u == this -> source)
	{
		sourceEdges[v] -= diff;
		if (v == this -> sink)
			sinkEdges[u] += diff;
		else
			for (i = 0; i < 10; i++)
				if (edges[v][i].v == u)
					edges[v][i].capacity += diff;
	}
	else if (u == this -> sink)
	{
		sinkEdges[v] -= diff;
		if (v == this -> source)
			sourceEdges[v] += diff;
		else
			for (i = 0; i < 10; i++)
				if (edges[v][i].v == u)
					edges[v][i].capacity += diff;
	}
	else
	{
		edges[u][pos].capacity -= diff;
		if (v == this -> source)
			sourceEdges[u] += diff;
		else if (v == this -> sink)
			sinkEdges[u] += diff;
		else
			for (i = 0; i < 10; i++)
				if (edges[v][i].v == u)
					edges[v][i].capacity += diff;
	}
	markActive(v);
}

long long Graph::result()
{
	long long maxFlow = 0;

	initializePreflow();
	for (int i = 0; i < this -> V; i++)
		if (sourceEdges[i] >= 0)
			push(this -> source, i);
	long long count = 0;
	while (!activeVertices.empty())
	{
		int u = activeVertices.front();
		activeVertices.pop();
		active[u] = false;

		if (u == source || u == sink)
			continue;

		for (int i = 0; i < 10; i++)
			if (edges[u][i].capacity > 0)
				push(u, edges[u][i].v);

		if (excess[u] > 0)
		{
			if (labelCount[label[u]] == 1)
				gap(label[u]);
			else
				relabel(u);
		}
		count++;
	}
	return maxFlow = excess[sink];
}

int main()
{
	int i, j, n, m, x, y, z;
	list<int>::iterator iter;
	for (i = 0; i < FLATIMAGESIZE + 10; i++)
	{
		sourceEdges[i] = -1;
		sinkEdges[i] = -1;
	}
	cin >> n >> m;
	Graph g(n);
	g.setTerminals(0, n - 1);
	while (m--)
	{
		cin >> x >> y >> z;
		if (x != y && z > 0)
			g.addEdge(x - 1, y - 1, z);
	}
	cout << g.result() << '\n';
}
