#include <algorithm>
#include <iostream>
#include <limits.h>
#include <list>
#include <vector>

using namespace std;

// The Graph object, on which the max-flow algorithm is to be run
class Graph
{
public:
	// Number of vertices in the graph.
	int V;
	// Height associated with each node; flow is pushed from higher nodes to lower nodes.
	int *height;
	// Excess flow (incoming - outgoing) associated with each node.
	int *excessFlow;
	
	list<int> *adj;
	vector< pair < int, pair <int, int> > > *edges;

	Graph(int V);
	void addEdge(int u, int v, int capacity, int flow);

	void initializePreflow(int source);
	int pushFlow(int u);
	void relabelVertex(int u);
	int maxFlow(int source, int sink);
};

Graph::Graph(int V)
{
	this -> V = V;
	adj = new list<int> [V];
	edges = new vector < pair < int, pair <int, int> > > [V];
	height = new int[V];
	excessFlow = new int[V];

	for (int i = 0; i < V; i++)
	{
		height[i] = 0;
		excessFlow[i] = 0;
	}
}

void Graph::addEdge(int u, int v, int capacity, int flow = 0)
{
	adj[u].push_back(v);
	edges[u].push_back(make_pair(v, make_pair(capacity, flow)));
}

void Graph::initializePreflow(int source)
{
	height[source] = this -> V;
	for (int i = 0; i < edges[source].size(); i++)
	{
		int neighbourOfSource = edges[source][i].first;
		// For all neighbours of the source, the excess flow at these nodes is the capacity of the incoming edges from the source.
		edges[source][i].second.second = edges[source][i].second.first;
		// cout << edges[source][i].first << ' ' << edges[source][i].second.first << ' ' << edges[source][i].second.second << '\n';
		// The excess flow at a node is the sum of all incoming flows.
		excessFlow[neighbourOfSource] += edges[source][i].second.first;
		// Adding a saturated reverse edge in the residual graph.
		edges[neighbourOfSource].push_back(make_pair(source, make_pair(0, -edges[source][i].second.first)));
	}
}

int Graph::pushFlow(int u)
{
	// Assuming excessFlow[u] > 0.
	int flowToPush = 0, vertexToBePushedInto = -1;
	for (int i = 0; i < edges[u].size(); i++)
	{
		// cout << "Here\n";
		if (edges[u][i].second.first > edges[u][i].second.second && height[edges[u][i].first] < height[u])
		{
			vertexToBePushedInto = edges[u][i].first;
			// cout << vertexToBePushedInto << '\n';
			flowToPush = min(excessFlow[u], edges[u][i].second.first - edges[u][i].second.second);
			excessFlow[u] -= flowToPush;
			excessFlow[vertexToBePushedInto] += flowToPush;
			edges[u][i].second.second += flowToPush;
			break;
		}
	}
	if (vertexToBePushedInto != -1)
		for (int i = 0; i < edges[vertexToBePushedInto].size(); i++)
			if (edges[vertexToBePushedInto][i].first == u)
				edges[vertexToBePushedInto][i].second.second -= flowToPush;
	// cout << vertexToBePushedInto << '\n';
	return vertexToBePushedInto;
}

void Graph::relabelVertex(int u)
{
	// Setting initial height to be updated to be maximum possible
	int minHeight = INT_MAX;
	// cout << minHeight << '\n';
	for (int i = 0; i < edges[u].size(); i++)
		// cout << edges[u][i].second.second << ' ' <<  edges[u][i].second.first << '\n';
		if (minHeight > height[edges[u][i].first] && edges[u][i].second.second < edges[u][i].second.first)
			minHeight = height[edges[u][i].first];
	// cout << minHeight << '\n';
	height[u] = minHeight + 1;
}

int Graph::maxFlow(int source, int sink)
{
	int i, flow = 0, V = this -> V;
	initializePreflow(source);
	for (int j = 0; j < V; j++)
		for (i = 0; i < V; i++)
			if (excessFlow[i] > 0 &&  i != sink)
				if (pushFlow(i) == -1)
					relabelVertex(i);
	for (i = 0; i < V; i++)
		flow = max(flow, excessFlow[i]);
	return flow;
}

int main()
{
	ios::sync_with_stdio(0);
	
	Graph g(6);
	g.addEdge(0, 1, 16);
	g.addEdge(0, 2, 13);
	g.addEdge(1, 2, 10);
	g.addEdge(2, 1, 4);
	g.addEdge(1, 3, 12);
	g.addEdge(2, 4, 14);
	g.addEdge(3, 2, 9);
	g.addEdge(3, 5, 20);
	g.addEdge(4, 3, 7);
	g.addEdge(4, 5, 4);

	int source = 0, sink = 5;
	cout << "Maximum flow in the network is " << g.maxFlow(source, sink) << '\n';
	return 0;
}