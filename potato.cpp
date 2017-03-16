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
	// Adjacency matrices
	ll **graph_flow;
	ll **graph_weights;
	// The constructor for the Graph object.
	Graph(int V);
	// Adding an edge to the corresponding vector with the given flow and capacity.
	void addEdge(int start, int end, ll capacity, ll flow);
	// Initializes the preflow of the graph.
	void initializePreflow(int source);
	// Returns the maximum flow in the graph.
	ll MF(int source, int sink);
	// Pushes flow in given edge.
	int pushFlow(int start, int end);
	// Relabels start to allow it to push flow through to one of its neighbours.
	void relabelVertex(int start);
};

Graph::Graph(int V)
{
	this -> V = V;
	graph_flow = new ll*[V];
	graph_weights = new ll*[V];
	for(int i=0; i<V; i++){
		graph_flow[i] = new ll[V];
		graph_weights[i] = new ll[V];
	}
	excessFlow = new ll[V];
	height = new ll[V];
	isActive = new bool[V];
	for(int i=0; i < V; i++){
		for(int j=0; j < V; j++){
			graph_weights[i][j] = 0;
			graph_flow[i][j] = 0;
		}
	}
}

void Graph::addEdge(int start, int end, ll capacity, ll flow = 0)
{
	graph_weights[start][end] = capacity;
	graph_flow[start][end] = flow;
}

void Graph::initializePreflow(int source)
{
	for (int i = 0; i < this -> V; i++)
	{
		height[i] = 0;
		excessFlow[i] = 0;
		isActive[i] = false;
	}
	for (int i = 0; i < V; i++)
	{
		if(graph_weights[source][i]){
			graph_flow[source][i] = graph_weights[source][i];
			excessFlow[i] = graph_weights[source][i];
		}
	}
	for (int i = 0; i < V; i++)
	{
		if(graph_weights[i][source]){
			graph_flow[i][source] = - graph_flow[source][i];
		}
	}
	height[source] = this -> V;
}

void Graph::relabelVertex(int start)
{
	ll minNeighbourHeight = LLONG_MAX;
	for (int i = 0; i < V; i++){
		if (graph_weights[start][i]){
			if (graph_weights[start][i] > graph_flow[start][i]){
				minNeighbourHeight = min(minNeighbourHeight, height[i]);
				height[start] = minNeighbourHeight + 1;
			}
		}
	}
}

int Graph::pushFlow(int start, int end)
{
	ll delta = min(excessFlow[start], graph_weights[start][end] - graph_flow[start][end]);
	graph_flow[start][end] += delta;
	graph_flow[end][start] -= delta;
	excessFlow[start] -= delta;
	excessFlow[end] += delta;
}

ll Graph::MF(int source, int sink)
{
	ll flow = 0;
	initializePreflow(source);

	for (int i = 0; i < this -> V; i++){
		if (height[i] < V && excessFlow[i] > 0 && !isActive[i] && i != source && i != sink){
			isActive[i] = true;
			activeNodes.push(i);
		}
	}

	int vertexToFix;
	while (!activeNodes.empty())
	{
		cout<<activeNodes.front()<<endl;
		vertexToFix = activeNodes.front();
		activeNodes.pop();
		isActive[vertexToFix] = false;

		bool break_out;
		while(excessFlow[vertexToFix] > 0 && height[vertexToFix] < V){
			break_out = true;
			for(int i = 0; i < V; i++){
				if(graph_weights[vertexToFix][i] > 0){
					if( !isActive[i] && graph_flow[vertexToFix][i] > 0 && height[vertexToFix] == height[i]+1){
						pushFlow(vertexToFix, i);
						break_out = false;
						activeNodes.push(i);
						isActive[i] = true;
					}
				}	
			}
			if(break_out){
				break;
			}		
		}
		if (excessFlow[vertexToFix] > 0 && height[vertexToFix] < V)
		{
			relabelVertex(vertexToFix);
			activeNodes.push(vertexToFix);
			isActive[vertexToFix] = true;
		}
	}
	flow = excessFlow[sink];
	return flow;
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
	cout<<"Called?\n";
	cout << g.MF(0, n - 1) << endl;
	return 0;
}
