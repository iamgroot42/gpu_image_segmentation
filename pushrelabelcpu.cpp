#include <algorithm>
#include <iostream>
#include <limits.h>
#include <queue>
#include <vector>

#define ll long long
using namespace std;

class Graph
{
public:
	int V;
	ll *height, *excessFlow;
	bool *isActive;
	queue<int> activeNodes;
	vector < pair <int, pair <ll, ll > > > *edgeSet;

	Graph(int V);
	void addEdge(int start, int end, ll capacity, ll flow);
	void BFS(int sink);

	void initializePreflow(int source, int sink);
	ll maxFlow(int source, int sink);
	ll pushFlow(int start);
	void relabelVertex(int start);
	int VertexToFix(int sink, int source);
};

Graph::Graph(int V)
{
	this -> V = V;
	height = new ll[V];
	excessFlow = new ll[V];
	isActive = new bool[V];
	edgeSet = new vector < pair <int, pair <ll, ll > > > [V];
}

void Graph::addEdge(int start, int end, ll capacity, ll flow = 0)
{
	edgeSet[start].push_back(make_pair(end, make_pair(capacity, flow)));
	// Uncomment the next line to get TLE, because of inefficiency of current implementation.
	edgeSet[end].push_back(make_pair(start, make_pair(capacity, flow)));
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
	for (int i = 0; i < V; i++)
	{
		height[i] = -1;
		excessFlow[i] = 0;
		isActive[i] = false;
	}
	BFS(sink);
	for (int i = 0; i < edgeSet[source].size(); i++)
	{
		edgeSet[source][i].second.second = edgeSet[source][i].second.first;
		excessFlow[edgeSet[source][i].first] += edgeSet[source][i].second.first;
		edgeSet[edgeSet[source][i].first].push_back(make_pair(source, make_pair(0, -1 * edgeSet[source][i].second.second)));
	}
	for (int i = 0; i < V; i++)
		if (excessFlow[i] > 0 && !isActive[i])
		{
			isActive[i] = true;
			activeNodes.push(i);
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
		if (height[edgeSet[start][i].first] < minNeighbourHeight && edgeSet[start][i].second.first > edgeSet[start][i].second.second)
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
	initializePreflow(source, sink);
	// int vertexToFix = VertexToFix(source, sink);
	int vertexToFix = -1;
	// while (vertexToFix != INT_MAX)
	while (!activeNodes.empty())
	{
		vertexToFix = activeNodes.front();
		activeNodes.pop();
		isActive[vertexToFix] = false;
		if (vertexToFix == source || vertexToFix == sink)
			continue;
		int neighbourToPushTo = pushFlow(vertexToFix);
		if (neighbourToPushTo == INT_MAX)
		{
			relabelVertex(vertexToFix);
			activeNodes.push(vertexToFix);
			isActive[vertexToFix]= true;
		}
		else
			if (excessFlow[neighbourToPushTo] > 0 && !isActive[neighbourToPushTo])
			{
				activeNodes.push(neighbourToPushTo);
				isActive[neighbourToPushTo] = true;
			}
		if (excessFlow[vertexToFix] > 0)
		{
			activeNodes.push(vertexToFix);
			isActive[vertexToFix] = true;
		}
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