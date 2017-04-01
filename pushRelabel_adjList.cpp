#include <bits/stdc++.h>

using namespace std;

long long edges[5001][5001];
long long sourceEdges[5001], sinkEdges[5001];

class Graph
{
public:
	bool *active, *reachable;
	int V, *labelCount, *label;
	long long *excess;
	queue<int> activeVertices;
	list<int> *adj;
	vector < pair<int, int> > cutEdges;

	Graph(int V);
	void addEdge(int u, int v, long long capacity, int source, int sink);
	void BFS(int source, int sink);

	void initializePreflow(int source);
	void markActive(int u);
	void relabel(int u, int source, int sink);
	void push(int u, int v, int source, int sink);
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
	adj = new list<int> [V];
	reachable = new bool[V];
}

void Graph::addEdge(int u, int v, long long capacity, int source, int sink)
{
	adj[u].push_back(v);

	if (u == source)
	{
		if (sourceEdges[v] < 0)
			sourceEdges[v] = capacity;
		else
			sourceEdges[v] += capacity;
	}
	else if (u == sink)
	{
		if (sinkEdges[v] < 0)
			sinkEdges[v] = capacity;
		else
			sinkEdges[v] += capacity;
	}
	else
	{
		if (edges[u][v] < 0)
			edges[u][v] = capacity;
		else
			edges[u][v] += capacity;
	}
}

void Graph::BFS(int source, int sink)
{
	list<int>::iterator iter;
	queue<int> reachableVertices;

	reachable[source] = true;
	reachableVertices.push(source);
	while (!reachableVertices.empty())
	{
		int x = reachableVertices.front(), y;
		reachableVertices.pop();

		for (iter = adj[x].begin(); iter != adj[x].end(); iter++)
		{
			y = *iter;
			if (!reachable[y])
			{
				if ( (x == source && sourceEdges[y] > 0) || (x == sink && sinkEdges[y] > 0) || (x != source && x != sink && edges[x][y] > 0))
				reachable[y] = true;
				reachableVertices.push(y);
			}
		}
	}
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
		reachable[i] = false;
	}
	label[source] = this -> V;
	labelCount[0] = this -> V - 1;
	labelCount[this -> V] = 1;
	for (int i = 0; i < this -> V; i++)
		if (sourceEdges[i] >= 0)
			excess[source] += sourceEdges[i];
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

void Graph::relabel(int u, int source, int sink)
{
	labelCount[label[u]]--;
	int minLabel = INT_MAX;
	for (int i = 0; i < this -> V; i++)
	{
		if ( (u == source && sourceEdges[i] > 0) || (u == sink && sinkEdges[i] > 0) || (u != source && u != sink && edges[u][i] > 0) )
		{
			minLabel = min(minLabel, label[i]);
			label[u] = minLabel + 1;
		}
	}
	labelCount[label[u]]++;
	markActive(u);
}

void Graph::push(int u, int v, int source, int sink)
{
	long long diff = excess[u];
	if (u == source && sourceEdges[v] > 0)
		diff = min(diff, sourceEdges[v]);
	else if (u == sink && sinkEdges[v] > 0)
		diff = min(diff, sinkEdges[v]);
	else if (u != source && u != sink && edges[u][v] > 0)
		diff = min(diff, edges[u][v]);
	if (diff == 0 || label[u] <= label[v])
		return;

	excess[u] -= diff;
	excess[v] += diff;

	if (u == source)
	{
		sourceEdges[v] -= diff;
		if (v == sink)
			sinkEdges[u] += diff;
		else
			edges[v][u] += diff;
	}
	else if (u == sink)
	{
		sinkEdges[v] -= diff;
		if (v == source)
			sourceEdges[v] += diff;
		else
			edges[v][u] += diff;
	}
	else
	{
		edges[u][v] -= diff;
		edges[v][u] += diff;
	}
	markActive(v);
}

long long Graph::result(int source, int sink)
{
	long long maxFlow = 0;
	initializePreflow(source);
	for (int i = 0; i < this -> V; i++)
		if (sourceEdges[i] >= 0)
			push(source, i, source, sink);
	while (!activeVertices.empty())
	{
		int u = activeVertices.front();
		activeVertices.pop();
		active[u] = false;

		if (u == source || u == sink)
			continue;

		for (int i = 0; i < this -> V && excess[u] > 0; i++)
			if (edges[u][i] > 0)
				push(u, i, source, sink);
		if (excess[u] > 0)
		{
			if (labelCount[label[u]] == 1)
				gap(label[u]);
			else
				relabel(u, source, sink);
		}
	}
	for (int i = 0; i < this -> V; i++)
		if (sourceEdges[i] > 0)
			maxFlow += sourceEdges[i];
	return maxFlow = max(maxFlow, excess[sink]);
}

int main()
{
	int i, j, n, m, x, y, z;
	list<int>::iterator iter;
	for (i = 0; i < 5001; i++)
	{
		sourceEdges[i] = -1;
		sinkEdges[i] = -1;
		for (j = 0; j < 5001; j++)
			edges[i][j] = -1;
	}
	cin >> n >> m;
	Graph g(n);
	while (m--)
	{
		cin >> x >> y >> z;
		if (x != y && z > 0)
		{
			g.addEdge(x - 1, y - 1, z, 0, n - 1);
			g.addEdge(y - 1, x - 1, z, 0, n - 1);
		}
	}
	cout << g.result(0, n - 1) << '\n';
	// g.BFS(0, n - 1);
	// for (int i = 0; i < n; i++)
	// 	if (g.reachable[i])
	// 		for (iter = g.adj[i].begin(); iter != g.adj[i].end(); iter++)
	// 			if (!g.reachable[*iter])
	// 				g.cutEdges.push_back(make_pair(i, *iter));
	// for (int i = 0; i < g.cutEdges.size(); i++)
	// 	cout << g.cutEdges[i].first << ' ' << g.cutEdges[i].second << '\n';
}
