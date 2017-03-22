#include <bits/stdc++.h>

using namespace std;

long long edges[5010][5010];

class Graph
{
public:
	bool *active;
	int V;
	int *labelCount, *label;
	long long *excess;
	queue<int> activeVertices;
	Graph(int V);
	void addEdge(int u, int v, long long capacity);

	void initializePreflow(int source);
	void markActive(int u);
	void relabel(int u);
	void push(int u, int v);
	void gap(int k);

	long long result(int source, int sink);
};

Graph::Graph(int V)
{
	this -> V = V;
	label = new int[V];
	labelCount = new int[2 * V];
	excess = new long long[V];
	active = new bool[V];
}

void Graph::addEdge(int u, int v, long long capacity)
{
	if (edges[u][v] < 0)
		edges[u][v] = capacity;
	else
		edges[u][v] += capacity;

	if (edges[v][u] < 0)
		edges[v][u] = capacity;
	else
		edges[v][u] += capacity;
}

void Graph::initializePreflow(int source)
{
	for (int i = 0; i < this -> V; i++)
	{
		active[i] = false;
		excess[i] = 0;
		label[i] = 0;
		labelCount[2 * i] = 0;
		labelCount[2 * i + 1] = 0;
	}
	label[source] = this -> V;
	labelCount[0] = this -> V - 1;
	labelCount[this -> V] = 1;
	for (int i = 0; i < this -> V; i++)
		if (edges[source][i] >= 0)
			excess[source] += edges[source][i];
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
	int minLabel = INT_MAX;
	for (int i = 0; i < this -> V; i++)
		if (edges[u][i] > 0)
		{
			minLabel = min(minLabel, label[i]);
			label[u] = minLabel + 1;
		}
	labelCount[label[u]]++;
	markActive(u);
}

void Graph::push(int u, int v)
{
	long long diff = min(excess[u], edges[u][v]);
	if (diff == 0 || label[u] <= label[v])
		return;
	excess[u] -= diff;
	excess[v] += diff;
	edges[u][v] -= diff;
	edges[v][u] += diff;
	markActive(v);
}

long long Graph::result(int source, int sink)
{
	long long maxFlow = 0;
	initializePreflow(source);
	for (int i = 0; i < this -> V; i++)
		if (edges[source][i] >= 0)
			push(source, i);
	while (!activeVertices.empty())
	{
		int u = activeVertices.front();
		activeVertices.pop();
		active[u] = false;

		if (u == source || u == sink)
			continue;

		for (int i = 0; i < this -> V && excess[u] > 0; i++)
			if (edges[u][i] > 0)
				push(u, i);
		if (excess[u] > 0)
		{
			if (labelCount[label[u]] == 1)
				gap(label[u]);
			else
				relabel(u);
		}
	}
	return (maxFlow = excess[sink]);
}

int main()
{
	int i, j, n, m, x, y, z;
	for (i = 0; i < 5010; i++)
		for (j = 0; j < 5010; j++)
			edges[i][j] = -1;
	for (i = 0; i < 5010; i++)
		for (j = 0; j < 5010; j++)
			edges[i][j] = -1;
	cin >> n >> m;
	Graph g(n);
	while (m--)
	{
		cin >> x >> y >> z;
		if (x != y)
			g.addEdge(x - 1, y - 1, z);
	}
	cout << g.result(0, n - 1) << '\n';
}
