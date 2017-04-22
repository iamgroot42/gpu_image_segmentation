#include <bits/stdc++.h>

#define maxNeighbours 10
#define imageSize 262 * 197
using namespace std;

struct Edge
{
	int v;
	long long capacity, flow;
	Edge(int v = -1, long long capacity = -1, long long flow = -1)
	{
		this -> v = v;
		this -> capacity = capacity;
		this -> flow = flow;
	}
};

vector <Edge> sourceEdges, sinkEdges;
Edge edges[imageSize][10];

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

void addEdgeAux(int u, int v, long long capacity, vector <Edge> &edgeSet)
{
	bool flag = false;
	for (int i = 0; i < edgeSet.size(); i++)
		if (edgeSet[i].v == v)
		{
			edgeSet[i].capacity += capacity;
			flag = true;
			break;
		}
	if (!flag)
		edgeSet.push_back(Edge(v, capacity, 0));
}

void pushAux1(long long *diff, int *pos, int v, vector <Edge> &edgeSet)
{
	int i;
	for (i = 0; i < edgeSet.size(); i++)
		if (edgeSet[i].v == v)
		{
			*pos = i;
			*diff = min(*diff, edgeSet[i].capacity - edgeSet[i].flow);
		}
}

bool BFSAux(int y, vector<Edge> & edgeSet)
{
	bool flag = true;
	for (int i = 0; i < edgeSet.size(); i++)
		if (edgeSet[i].v == y && edgeSet[i].capacity > edgeSet[i].flow && edgeSet[i].flow > 0)
		{
			cout << edgeSet[i].capacity << ' ' <<  edgeSet[i].flow << '\n';
			flag = true;
		}
	return flag;
}

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
		addEdgeAux(u, v, capacity, sourceEdges);
	else if (u == this -> sink)
		addEdgeAux(u, v, capacity, sinkEdges);
	else
		for (int i = 0; i < maxNeighbours; i++)
		{
			if (edges[u][i].v == v)
			{
				edges[u][i].capacity += capacity;
				break;
			}
			else if (edges[u][i].capacity == -1)
			{
				edges[u][i].v = v;
				edges[u][i].capacity = capacity;
				edges[u][i].flow = 0;
				break;
			}
		}
}

void Graph::BFS()
{
	queue<int> neighbours;
	list<int>::iterator iter;
	reachable[this -> source] = true;
	neighbours.push(this -> source);
	while (!neighbours.empty())
	{
		int x = neighbours.front(), y, i;
		neighbours.pop();
		for (iter = adj[x].begin(); iter != adj[x].end(); iter++)
		{
			y = *iter;
			if (!reachable[y])
			{
				if ((x == this -> source && BFSAux(y, sourceEdges)) || (x == this -> sink && BFSAux(y, sinkEdges)))
					{
						reachable[y] = true;
						neighbours.push(y);
					}
				else if (x != this -> source && x != this -> sink)
					for (i = 0; i < maxNeighbours; i++)
						if (edges[x][i].v == y && edges[x][i].capacity > edges[x][i].flow && edges[x][i].flow > 0)
						{
							reachable[y] = true;
							neighbours.push(y);
						}
			}
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
		reachable[i] = false;
	}
	label[this -> source] = this -> V;
	labelCount[0] = this -> V - 1;
	labelCount[this -> V] = 1;
	for (int i = 0; i < sourceEdges.size(); i++)
		excess[this -> source] += sourceEdges[i].capacity;
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
	for (i = 0; i < maxNeighbours; i++)
		if (edges[u][i].capacity > edges[u][i].flow)
		{
			assert(edges[u][i].capacity > 0);
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

	if (u == this -> source)
		pushAux1(&diff, &pos, v, sourceEdges);
	else if (u == this -> sink)
		pushAux1(&diff, &pos, v, sinkEdges);
	else
		for (i = 0; i < maxNeighbours; i++)
			if (edges[u][i].v == v && edges[u][i].capacity > edges[u][i].flow)
			{
				pos = i;
				diff = min(diff, edges[u][i].capacity - edges[u][i].flow);
			}

	if (diff == 0 || label[u] <= label[v])
	{
		// cout << "Returning\n";
		return;
	}
	excess[u] -= diff;
	excess[v] += diff;

	if (u == this -> source)
	{
		sourceEdges[pos].flow += diff;
		if (v == this -> sink)
			for (int i = 0; i < sinkEdges.size(); i++)
				if (sinkEdges[i].v == u)
					sinkEdges[i].flow -= diff;
		else
			for (i = 0; i < maxNeighbours; i++)
				if (edges[v][i].v == u)
					edges[v][i].flow -= diff;
	}
	else if (u == this -> sink)
	{
		sinkEdges[pos].flow += diff;
		if (v == this -> source)
			for (int i = 0; i < sourceEdges.size(); i++)
				if (sourceEdges[i].v == u)
					sourceEdges[i].flow -= diff;
		else
			for (i = 0; i < maxNeighbours; i++)
				if (edges[v][i].v == u)
					edges[v][i].flow -= diff;
	}
	else
	{
		edges[u][pos].flow += diff;
		for (i = 0; i < maxNeighbours; i++)
			if (edges[v][i].v == u)
				edges[v][i].flow -= diff;
	}
	markActive(v);
}

long long Graph::result()
{
	long long maxFlow = 0;

	initializePreflow();
	for (int i = 0; i < sourceEdges.size(); i++)
		if (sourceEdges[i].capacity > sourceEdges[i].flow)
			push(this -> source, sourceEdges[i].v);
	while (!activeVertices.empty())
	{
		// cout << activeVertices.front() << ' ' << excess[activeVertices.front()] << '\n';
		int u = activeVertices.front();
		activeVertices.pop();
		active[u] = false;

		if (u == source || u == sink)
			continue;

		for (int i = 0; i < maxNeighbours; i++)
			if (edges[u][i].capacity > edges[u][i].flow)
				push(u, edges[u][i].v);
		if (excess[u] > 0)
		{
			bool temp = false;
			for (int i = 0; i < maxNeighbours; i++)
				if (label[u] <= label[edges[u][i].v] && edges[u][i].capacity > edges[u][i].flow)
					temp = true;
			if (!temp)
				continue;
			if (labelCount[label[u]] == 1)
				gap(label[u]);
			else
				relabel(u);
		}
	}
	return maxFlow = excess[sink];
}

int main()
{
	clock_t begin = clock();
	int i, j, n, m, x, y, z;
	list<int>::iterator iter;
	cin >> n >> m;
	Graph g(n);
	g.setTerminals(0, n - 1);
	while (m--)
	{
		cin >> x >> y >> z;
		if (x != y)
			g.addEdge(x - 1, y - 1, z);
	}
	// cout << g.result() << '\n';
	g.result();
	// clock_t end = clock();
	// double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	// cout << elapsed_secs << '\n';
	g.BFS();
	for (i = 0; i < n; i++)
		if (g.reachable[i])
			cout << i << '\n';
		// for (iter = g.adj[i].begin(); iter != g.adj[i].end(); iter++)
		// 	if (g.reachable[i] && !g.reachable[*iter])
		// 		if (i != 0 && *iter <= n - 3)
		// 			cout << i << ' ' << *iter << '\n';
}
