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
	int V;
	ll *height, *excessFlow, *residual;
	bool *isActive;
	queue<int> activeNodes;
	ll **graph_flow;
	ll **graph_weights;
	Graph(int V);
	void addEdge(int start, int end, ll capacity);
	void initializePreflow(int source);
	ll MF(int source, int sink);
	int pushFlow(int start, int end);
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
	height = new ll[V];
	excessFlow = new ll[V];
	isActive = new bool[V];
	for(int i=0; i < V; i++){
		for(int j=0; j < V; j++){
			graph_weights[i][j] = 0;
			graph_flow[i][j] = 0;
		}
	}
}

void Graph::addEdge(int start, int end, ll capacity)
{
	graph_weights[start][end] = capacity;
}

void Graph::initializePreflow(int source)
{
	for(int i = 0; i < V; i++){
		height[i] = 0;
		excessFlow[i] = 0;
		isActive[i] = false;
	}
	height[source] = V;
	for (int i = 0; i < V; i++)
	{
		if(graph_weights[source][i]){
			graph_flow[source][i] = graph_weights[source][i];
			graph_flow[i][source] = -graph_weights[source][i];
			excessFlow[i] = graph_weights[source][i];
			excessFlow[source] -= graph_weights[source][i];
		}	
	}
}

void Graph::relabelVertex(int start)
{
	ll temp = -1;
	for (int i = 0; i < V; i++){
		if(graph_weights[start][i]){
			if(graph_weights[start][i] > graph_flow[start][i]){
				if(temp == -1 || temp > height[i]){
					temp = height[i];
				}
			}
		}
	}
	height[start] = 1 + temp;
}

int Graph::pushFlow(int start, int end)
{
	ll delta = min(excessFlow[start], graph_weights[start][end] - graph_flow[start][end]);
	graph_flow[start][end] += delta;
	graph_flow[end][start] = -graph_flow[start][end];
	excessFlow[start] -= delta;
	excessFlow[end] += delta;
}

ll Graph::maxFlow(int source, int sink)
{
	ll m;
	initializePreflow(source);

	for (int i = 0; i < V; i++){
		if(graph_weights[source][i] && i != sink){
			activeNodes.push(i);
			isActive[i] = true;
		}
	}

	int vertexToFix;
	while (!activeNodes.empty()){
		vertexToFix = activeNodes.front();
		m = -1;

		for (int i = 0; i < V && excessFlow[vertexToFix] > 0; i++){
			if(graph_weights[vertexToFix][i]){
				if(graph_weights[vertexToFix][i] > graph_flow[vertexToFix][i]){
					if(height[vertexToFix] > height[i]){
						pushFlow(vertexToFix, i);
						if(!isActive[i] && i != source && i != sink){
							isActive[i] = true;
							activeNodes.push(i);
						}
					}
					else if(m == -1){
						m = height[i];
					}
					else{
						m = min(m, height[i]);
					}
				}
			}
		}

		if(excessFlow[vertexToFix]){
			height[vertexToFix] = 1 + m;
		}
		else{
			isActive[vertexToFix] = false;
			activeNodes.pop();
		}
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
		if (x != y)	// Not handling self loops as flow does not change, and undirected graph.
		{
			g.addEdge(x - 1, y - 1, z);
			g.addEdge(y - 1, x - 1, z);
		}
		else{
			g.addEdge(x - 1, y - 1, z);
		}
	}
	cout << g.maxFlow(0, n - 1) << endl;
	return 0;
}
