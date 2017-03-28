#include <bits/stdc++.h>

using namespace std;

map <int, int> edges[5010];
map <int, int>::iterator iter;

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
	void addEdge(int u, int v, long long capacity);
	void BFS(int source);

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
	adj = new list<int> [V];
	reachable = new bool[V];
}

void Graph::addEdge(int u, int v, long long capacity)
{
	adj[u].push_back(v);
	adj[v].push_back(u);

	if (edges[u].count(v) == 0)
		edges[u][v] = capacity;
	else
		edges[u][v] += capacity;

	if (edges[v].count(u) == 0)
		edges[v][u] = capacity;
	else
		edges[v][u] += capacity;
}

void Graph::BFS(int source)
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
			if (!reachable[y] && edges[x][y] > 0)
			{
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

	for (iter = edges[source].begin(); iter != edges[source].end(); iter++)
		excess[source] += edges[source][iter -> first];
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
	for (iter = edges[u].begin(); iter != edges[u].end(); iter++)
		if (edges[u][iter -> first] > 0)
		{
			minLabel = min(minLabel, label[iter -> first]);
			label[u] = minLabel + 1;
		}
	labelCount[label[u]]++;
	markActive(u);
}

void Graph::push(int u, int v)
{
	long long diff = min(excess[u], (long long)(edges[u][v]));
	if (diff == 0 || label[u] <= label[v])
		return;
	excess[u] -= diff;
	excess[v] += diff;
	edges[u][v] -= diff;
	edges[v][u] += diff;
	markActive(v);
}

void Graph::markActive(int u)
{
	if (!active[u] && excess[u] > 0)
	{
		active[u] = true;
		activeVertices.push(u);
	}
}

long long Graph::result(int source, int sink)
{
	long long maxFlow = 0;
	initializePreflow(source);
	for (iter = edges[source].begin(); iter != edges[source].end(); iter++)
		push(source, iter -> first);
	while (!activeVertices.empty())
	{
		int u = activeVertices.front();
		activeVertices.pop();
		active[u] = false;

		if (u == source || u == sink)
			continue;

		for (iter = edges[u].begin(); iter != edges[u].end() && excess[u] > 0; iter++)
			if (edges[u][iter -> first] > 0)
				push(u, iter -> first);

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
	cin >> n >> m;
	Graph g(n);
	while (m--)
	{
		cin >> x >> y >> z;
		if (x != y && z)
			g.addEdge(x - 1, y - 1, z);
	}
	cout << g.result(0, n - 1) << '\n';
	g.BFS(0);
	for (int i = 0; i < n; i++)
		if (g.reachable[i])
			for (iter = g.adj[i].begin(); iter != g.adj[i].end(); iter++)
				if (!g.reachable[*iter])
					g.cutEdges.push_back(make_pair(i, *iter));
	bool segmentation[n] = {false};
	for (int i = 0; i < g.cutEdges.size(); i++){
		// cout << g.cutEdges[i].first << ' ' << g.cutEdges[i].second << '\n';
		if(g.cutEdges[i].first == n-1 || g.cutEdges[i].second == n-1){
			if(g.cutEdges[i].first == n-1){
				segmentation[g.cutEdges[i].second] = true;
			}
			else{
				segmentation[g.cutEdges[i].first] = true;	
			}
		}
	}
	cout<<"Segmentation vector:"<<endl;
	for(int i=0;i<n;i++){
		cout<<segmentation[i]<<" ";
	}
	cout<<endl;
	// n = -1;
	// int count = 1;
	// while (n)
	// {
	// 	cin >> n;
	// 	if (!n)
	// 		break;
	// 	for (i = 0; i < 5010; i++)
	// 		edges[i].clear();
	// 	Graph g(n);
	// 	int source, sink;
	// 	cin >> source >> sink >> m;
	// 	source--;
	// 	sink--;
	// 	while (m--)
	// 	{
	// 		cin >> x >> y >> z;
	// 		g.addEdge(x - 1, y - 1, z);
	// 	}
	// 	cout << "Network " << count++ << '\n';
	// 	cout << "The bandwidth is " << g.result(source, sink) << ".\n\n";
	// }
}