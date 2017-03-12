#include <algorithm>
#include <iostream>
#include <limits.h>
#include <vector>

#define ll long long
using namespace std;

class Graph
{
public:
	int V;
	vector < pair <int, pair <ll, ll > > > *edgeSet;
	ll *height, *excessFlow;

	Graph(int V);
	void addEdge(int start, int end, ll capacity, ll flow);

	void initializePreflow(int source);
	int VertexToFix(int sink, int source);
	void relabelVertex(int start);
	ll pushFlow(int start);
	ll maxFlow(int source, int sink);
};

Graph::Graph(int V)
{
	this -> V = V;
	height = new ll[V];
	excessFlow = new ll[V];
	edgeSet = new vector < pair <int, pair <ll, ll > > > [V];
}

void Graph::addEdge(int start, int end, ll capacity, ll flow = 0)
{
	edgeSet[start].push_back(make_pair(end, make_pair(capacity, flow)));
}

void Graph::initializePreflow(int source)
{
	for (int i = 0; i < V; i++)
	{
		height[i] = 0;
		excessFlow[i] = 0;
	}
	height[source] = this -> V;
	for (int i = 0; i < edgeSet[source].size(); i++)
	{
		edgeSet[source][i].second.second = edgeSet[source][i].second.first;
		excessFlow[edgeSet[source][i].first] += edgeSet[source][i].second.first;
		edgeSet[edgeSet[source][i].first].push_back(make_pair(source, make_pair(0, -1 * edgeSet[source][i].second.second)));
	}
}

int Graph::VertexToFix(int source, int sink)
{
	for (int i = 0; i < V; i++)
		if (i != source && i != sink && excessFlow[i] > 0)
			return i;
	return INT_MAX;
}

void Graph::relabelVertex(int start)
{
	int minNeighbourHeight = INT_MAX;
	for (int i = 0; i < edgeSet[start].size(); i++)
		if (edgeSet[start][i].second.first > edgeSet[start][i].second.second && height[edgeSet[start][i].first] < minNeighbourHeight)
		{
			minNeighbourHeight = height[edgeSet[start][i].first];
			height[start] = minNeighbourHeight + 1;
		}
}

ll Graph::pushFlow(int start)
{
	int neighbourToPushTo = INT_MAX, flowToPush = 0;
	for (int i = 0; i < edgeSet[start].size(); i++)
	{
		if (edgeSet[start][i].second.first > edgeSet[start][i].second.second && height[edgeSet[start][i].first] == height[start] - 1 && excessFlow[start] > 0)
		{
			neighbourToPushTo = edgeSet[start][i].first;
			flowToPush = min(excessFlow[start], (ll)(edgeSet[start][i].second.first - edgeSet[start][i].second.second));
			excessFlow[start] -= flowToPush;
			excessFlow[neighbourToPushTo] += flowToPush;
			edgeSet[start][i].second.second += flowToPush;
			for (int j = 0; j < edgeSet[neighbourToPushTo].size(); j++)
				if (edgeSet[neighbourToPushTo][j].first == start)
				{
					edgeSet[neighbourToPushTo][j].second.second -= flowToPush;
					return neighbourToPushTo;
				}
			edgeSet[neighbourToPushTo].push_back(make_pair(start, make_pair(flowToPush, 0)));
			return neighbourToPushTo;
		}
	}
	return neighbourToPushTo;
}

ll Graph::maxFlow(int source, int sink)
{
	initializePreflow(source);
	int vertexToFix = VertexToFix(source, sink);
	while (vertexToFix != INT_MAX)
	{
		if (pushFlow(vertexToFix) == INT_MAX)
			relabelVertex(vertexToFix);
		vertexToFix = VertexToFix(source, sink);
	}
	return excessFlow[sink];
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
	cout << g.maxFlow(0, n - 1) << '\n';
	return 0;
}