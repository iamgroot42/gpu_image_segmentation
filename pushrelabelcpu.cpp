#include <algorithm>
#include <iostream>
#include <limits.h>
#include <list>
#include <queue>
#include <vector>

#define ll long long
using namespace std;

class Graph
{
public:
	// Number of nodes in the graph.
	int V;
	// The height(label) and excess of each node.
	ll *height, *excessFlow;
	// Maintains if a node is not the source/sink, and is overflowing.
	bool *isActive;
	// A queue of active nodes, used for O(1) FIFO.
	queue<int> activeNodes;
	// The set of edges in the graph, with each vector being the adjacency list of the ith node, and each entry having the flow
	// and capacity of the corresponding edge.
	vector < pair <int, pair < ll, ll > > > *edgeSet;
	// Adjacency list for current-arc implementation (Wiki)
	list<int> *adj;

	// The constructor for the Graph object.
	Graph(int V);
	// Adding an edge to the corresponding vector with the given flow and capacity.
	void addEdge(int start, int end, ll capacity, ll flow);
	// A simple implementation of BFS for setting the initial labels; performs slightly better by reducing number of
	// relabellings as compared to just setting the source to have a label of |V| and others 0.
	void BFS(int sink);

	// Initializes the preflow of the graph.
	void initializePreflow(int source, int sink);
	// Returns the maximum flow in the graph.
	ll maxFlow(int source, int sink);
	// Pushes flow from start to one of its neighbours.
	int pushFlow(int start);
	// Relabels start to allow it to push flow through to one of its neighbours.
	void relabelVertex(int start);
	// Discharges a vertex
	void dischargeVertex(int start);
};

Graph::Graph(int V)
{
	this -> V = V;

	adj = new list<int> [V];
	edgeSet = new vector < pair <int, pair < ll, ll > > > [V];
	excessFlow = new ll[V];
	height = new ll[V];
	isActive = new bool[V];
}

void Graph::addEdge(int start, int end, ll capacity, ll flow = 0)
{
	adj[start].push_back(end);
	edgeSet[start].push_back(make_pair(end, make_pair(capacity, flow)));
}

void Graph::BFS(int sink)
{
	bool *visited = new bool[V];
	queue<int> q;
	for (int i = 0; i < V; i++)
		visited[i] = false;
	height[sink] = 0;
	visited[sink] = true;
	q.push(sink);
	while (!q.empty())
	{
		int x = q.front();
		q.pop();
		for (int i = 0; i < edgeSet[x].size(); i++)
		{
			int neighbourToVisit = edgeSet[x][i].first;
			if (!visited[neighbourToVisit])
			{
				visited[neighbourToVisit] = true;
				height[neighbourToVisit] = height[x] + 1;
				q.push(neighbourToVisit);
			}
		}
	}
}

void Graph::initializePreflow(int source, int sink)
{
	for (int i = 0; i < this -> V; i++)
	{
		height[i] = -1;
		// height[i] = 0;
		excessFlow[i] = 0;
		isActive[i] = false;
	}
	BFS(sink);
	// height[source] = this -> V;
	for (int i = 0; i < edgeSet[source].size(); i++)
		excessFlow[source] += edgeSet[source][i].second.first;
}

void Graph::relabelVertex(int start)
{
	ll minNeighbourHeight = LLONG_MAX;
	for (int i = 0; i < edgeSet[start].size(); i++)
		if (edgeSet[start][i].second.first > edgeSet[start][i].second.second)
		{
			minNeighbourHeight = min(minNeighbourHeight, height[edgeSet[start][i].first]);
			height[start] = minNeighbourHeight + 1;
		}
}

int Graph::pushFlow(int start)
{
	int neighbourToPushTo = INT_MAX;
	ll flowToPush = 0;
	for (int i = 0; i < edgeSet[start].size(); i++)
		if (edgeSet[start][i].second.first > edgeSet[start][i].second.second && height[edgeSet[start][i].first] == height[start] - 1/* && excessFlow[start] > 0*/)
		{
			neighbourToPushTo = edgeSet[start][i].first;
			flowToPush = min(excessFlow[start], (ll)(edgeSet[start][i].second.first - edgeSet[start][i].second.second));
			excessFlow[start] -= flowToPush;
			excessFlow[neighbourToPushTo] += flowToPush;
			edgeSet[start][i].second.second += flowToPush;
			break;
		}
	if (neighbourToPushTo != INT_MAX)
	{
		for (int j = 0; j < edgeSet[neighbourToPushTo].size(); j++)
			if (edgeSet[neighbourToPushTo][j].first == start)
			{
				edgeSet[neighbourToPushTo][j].second.second -= flowToPush;
				return neighbourToPushTo;
			}
		edgeSet[neighbourToPushTo].push_back(make_pair(start, make_pair(0, -flowToPush)));
	}
	return neighbourToPushTo;
}

ll Graph::maxFlow(int source, int sink)
{
	ll flow = 0;
	initializePreflow(source, sink);
	while (excessFlow[source] > 0)
		if (pushFlow(source) == INT_MAX)
			relabelVertex(source);
	for (int i = 0; i < this -> V; i++)
		if (excessFlow[i] > 0 && !isActive[i] && i != source && i != sink)
		{
			isActive[i] = true;
			activeNodes.push(i);
		}
	int vertexToFix = -1;
	while (!activeNodes.empty())
	{
		vertexToFix = activeNodes.front();
		activeNodes.pop();
		isActive[vertexToFix] = false;
	
		if (vertexToFix == source || vertexToFix == sink)
			continue;
		int neighbourToPushTo = pushFlow(vertexToFix);
		while (neighbourToPushTo != INT_MAX)
		{
			if (!isActive[neighbourToPushTo] && neighbourToPushTo != source && neighbourToPushTo != sink)
			{
				isActive[neighbourToPushTo] = true;
				activeNodes.push(neighbourToPushTo);
			}
			if (excessFlow[vertexToFix] > 0)
				neighbourToPushTo = pushFlow(vertexToFix);
			else
				break;
		}
		if (excessFlow[vertexToFix] > 0)
		{
			relabelVertex(vertexToFix);
			activeNodes.push(vertexToFix);
			isActive[vertexToFix] = true;
		}
	}
	return (flow = excessFlow[sink]);
}

int main()
{
	int i, m, n, x, y, z;
	cin >> n >> m;
	Graph g(n);
	while (m--)
	{
		cin >> x >> y >> z;
		if (x != y)	// Not handling self loops as flow does not change, and undirected graph.
		{
			g.addEdge(x - 1, y - 1, z);
			g.addEdge(y - 1, x - 1, z);
		}
	}
	cout << g.maxFlow(0, n - 1) << '\n';
	return 0;
}