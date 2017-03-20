#include <assert.h>
#include <algorithm>
#include <iostream>
#include <limits.h>
#include <list>
#include <queue>
#include <vector>

#define ll long long
using namespace std;

struct Edge{
	ll capacity, flow, residue;
	int color;
	Edge(ll capacity=-1,ll flow=-1,ll residue=0,int color=-1)
	{
		this->capacity=capacity;
		this->flow=flow;
		this->residue=residue;
		this->color=color;
	}
};

class Graph{
public:
	int V;
	ll *height, *excessFlow, *residual;
	bool *isActive;
	queue<int> activeNodes;
	Edge **G;
	Graph(int V);
	void addEdge(int start, int end, ll capacity, bool color);
	void initializePreflow(int source);
	ll maxFlow(int source, int sink);
	int pushFlow(int start, int end);
	void relabelVertex(int start);
};

Graph::Graph(int V){
	this -> V = V;
	G = new Edge*[V];
	for(int i=0; i<V; i++){
		G[i] = new Edge[V];
	}
	for (i = 0; i < V; i++)
		for (j = 0; j < V; j++)
		{
			G[i][j].capacity = 0;
			G[i][j].flow = 0;
		}
	height = new ll[V];
	excessFlow = new ll[V];
	isActive = new bool[V];
}

void Graph::addEdge(int start, int end, ll capacity, bool color){
	graph_weights[start][end].capacity += capacity;
	graph_weights[start][end].color = color;
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

			graph_flow2[i][source] = graph_weights[source][i];
			graph_flow2[source][i] = -graph_weights[source][i];

			excessFlow[i] = graph_weights[source][i];
			excessFlow[source] -= graph_weights[source][i];
		}	
	}
	// for(int i=0; i<V; i++){
	// 	for(int j=0; j<V; j++){
	// 		if(graph_weights[i][j]){
	// 			res_graph[i][j] = graph_weights[i][j] - graph_flow[i][j];
	// 		}
	// 	}
	// }
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

	graph_flow2[end][start] -= delta;
	graph_flow2[start][end] = +graph_flow2[end][start];

	excessFlow[start] -= delta;
	excessFlow[end] += delta;
}

ll Graph::maxFlow(int source, int sink)
{
	ll m;
	initializePreflow(source);

	for(int i=0; i<V; i++){
		for(int j=0;j<V;j++){
			cout<<graph_flow[i][j]<<" ";
		}
		cout<<endl;
	}

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
			// cout << i << '\n';
			if(graph_weights[vertexToFix][i]){
				// if(graph_weights[vertexToFix][i] > graph_flow[vertexToFix][i]){
				if(graph_weights[vertexToFix][i] > graph_flow[i][vertexToFix]){
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
				// if(graph_weights[i][vertexToFix] > graph_flow2[i][vertexToFix]){
				// 	if(height[i] > height[vertexToFix]){
				// 		pushFlow(i, vertexToFix);
				// 		if(!isActive[i] && i != source && i != sink){
				// 			isActive[i] = true;
				// 			activeNodes.push(i);
				// 		}
				// 	}
				// 	else if(m == -1){
				// 		m = height[i];
				// 	}
				// 	else{
				// 		m = min(m, height[i]);
				// 	}
				// }
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
			g.addEdge(x - 1, y - 1, z, 0);
			g.addEdge(y - 1, x - 1, z, 1);
		}
		// else{
		// 	g.addEdge(x - 1, y - 1, z);
		// }
	}
	cout << g.maxFlow(0, n - 1) << endl;
	// int i, m, n, x, y, z, count = 1;
	// n = -1;
	// while (n)
	// {
	// 	cin >> n;
	// 	if (!n)
	// 		break;
	// 	Graph g(n);
	// 	int source, sink;
	// 	cin >> source >> sink >> m;
	// 	cout << source << ' ' << sink << '\n';
	// 	while (m--)
	// 	{
	// 		cin >> x >> y >> z;
	// 		if(x!=y){
	// 			g.addEdge(x - 1, y - 1, z);
	// 			g.addEdge(y - 1, x - 1, z);
	// 		}
	// 	}
	// 	cout << "Network " << count++ << '\n';
	// 	cout << "The bandwidth is " << g.maxFlow(source - 1, sink - 1) << ".\n\n";
	// }
	return 0;
}
