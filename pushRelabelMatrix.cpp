
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <limits.h>
#include <map>
#include <list>
#include <queue>
#include <vector>

using namespace std;

struct Edge
{
	long long capacity, flow;

	Edge(long long capacity = -1, long long flow = -1)
	{
		this -> capacity = capacity;
		this -> flow = flow;
	}
};

Edge edgeSet[5010][5010];

class Graph
{
public:
	int V;
	bool *isActive;
	long long *label, *excessFlow;
	queue<int> active;

	Graph(int V);
	void addEdge(int u, int v, long long capacity, long long flow);

	void init(int source, int sink);
	void relabel(int u);
	void push(int u, int v);
	void markActive(int u);
	long long maxFlow(int source, int sink);
};

Graph::Graph(int V)
{
	this -> V = V;

	isActive = new bool[V];
	label = new long long[V];
	excessFlow = new long long[V];
}

void Graph::addEdge(int u, int v, long long capacity, long long flow)
{
	if (edgeSet[u][v].capacity >= 0)
	{
		edgeSet[u][v].capacity += capacity;
		edgeSet[u][v].flow += flow;
	}
	else
		edgeSet[u][v] = Edge(capacity, flow);

	if (edgeSet[v][u].capacity >= 0)
	{
		edgeSet[v][u].capacity += capacity;
		edgeSet[v][u].flow += flow;
	}
	else
		edgeSet[v][u] = Edge(capacity, flow);
}

void Graph::init(int source, int sink)
{
	for (int i = 0; i < this -> V; i++)
	{
		label[i] = 0;
		excessFlow[i] = 0;
		isActive[i] = false;
		if (edgeSet[source][i].capacity >= 0)
		{
			excessFlow[i] = edgeSet[source][i].capacity;
			edgeSet[source][i].flow += edgeSet[source][i].capacity;
			edgeSet[i][source].flow += -1 * edgeSet[source][i].capacity;
		}
	}
	label[source] = this -> V;
}

void Graph::relabel(int u)
{
	long long minLabel = LLONG_MAX;
	for (int i = 0; i < this -> V; i++)
		if (edgeSet[u][i].capacity >= 0/* && edgeSet[u][i].flow >= 0*/ && edgeSet[u][i].capacity > edgeSet[u][i].flow)
		{
			minLabel = min(minLabel, label[i]);
			label[u] = minLabel + 1;
		}
	markActive(u);
}

void Graph::push(int u, int v)
{
	if (edgeSet[u][v].capacity < 0/* || edgeSet[u][v].flow < 0*/ || edgeSet[u][v].capacity == edgeSet[u][v].flow || label[u] <= label[v])
		return;
	assert(label[u] == label[v] + 1);
	long long flow = min(excessFlow[u], (long long)(edgeSet[u][v].capacity - edgeSet[u][v].flow));
	edgeSet[u][v].flow += flow;
	edgeSet[v][u].flow -= flow;
	excessFlow[u] -= flow;
	excessFlow[v] += flow;
	markActive(v);
}

void Graph::markActive(int u)
{
	if (!isActive[u] && excessFlow[u] > 0)
	{
		isActive[u] = true;
		active.push(u);
	}
}

long long Graph::maxFlow(int source, int sink)
{
	int u;
	long long ans = 0;
	init(source, sink);
	for (int i = 0; i < this -> V; i++)
		markActive(i);
	while (!active.empty())
	{
		u = active.front();
		isActive[u] = false;
		active.pop();
		if (u == source || u == sink)
			continue;
		for (int i = 0; i < this -> V && excessFlow[u] > 0; i++)
			push(u, i);
		if (excessFlow[u] > 0)
			relabel(u);
	}
	return (ans = excessFlow[sink]);
}

int main()
{
	int i, m, n, x, y, z, count = 1;
	// n = -1;
	// while (n)
	// {
	// 	cin >> n;
	// 	if (!n)
	// 		break;
	// 	Graph g(n);
	// 	int source, sink;
	// 	cin >> source >> sink >> m;
	// 	while (m--)
	// 	{
	// 		cin >> x >> y >> z;
	// 		g.addEdge(x - 1, y - 1, z, 0);
	// 	}
	// 	cout << "Network " << count++ << '\n';
	// 	cout << "The bandwidth is " << g.maxFlow(source - 1, sink - 1) << ".\n\n";
	// }
	cin >> n >> m;
	Graph g(n);
	while (m--)
	{
		cin >> x >> y >> z;
		if (x != y)
			g.addEdge(x - 1, y - 1, z, 0);
	}
	cout << g.maxFlow(0, n - 1) << '\n';
	return 0;
}