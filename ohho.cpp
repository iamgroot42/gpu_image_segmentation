#include <assert.h>
#include <algorithm>
#include <iostream>
#include <queue>
#include <utility>

using namespace std;

struct Edge{
	long long capacity, flow;
	int color;
	Edge(long long capacity = 0, long long flow = -1, int color = -1){
		this -> capacity = capacity;
		this -> flow = flow;
		this -> color = color;
	}
};

pair <Edge,Edge> edges[5010][5010];

class Graph{
public:
	int V;
	long long *label, *excessFlow;
	bool *active;
	queue<int> activeNodes;

	Graph(int V);
	void addEdge(int u, int v, 	long long capacity);
	void initializePreflow(int source);
	void push(int s, int t, int color);
	void relabel(int v);
	long long maxFlow(int source, int sink);
};

Graph::Graph(int V){
	this -> V = V;
	label = new long long[V];
	excessFlow = new long long[V];
	active = new bool[V];
	for(int i=0; i<V; i++){
		for(int j=0; j<V;j++){
			edges[i][j].first.flow = 0;
			edges[i][j].second.flow = 0;
			edges[i][j].first.capacity = -1;
			edges[i][j].second.capacity = 0;
		}
	}
}

void Graph::addEdge(int u, int v, long long capacity){
	if(edges[u][v].first.capacity == -1){
		edges[u][v].first.capacity = 0;
	}
	edges[u][v].first.capacity += capacity;
	edges[u][v].first.flow = 0;
	edges[u][v].first.color = 1;

	// edges[v][u].second.capacity += 0;
	edges[v][u].second.flow = 0;
	edges[v][u].second.color = 2;
}

void Graph::initializePreflow(int source){
	for (int i = 0; i < this -> V; i++){
		label[i] = 0;
		excessFlow[i] = 0;
		active[i] = false;
	}
	label[source] = this -> V;
	for (int i = 0; i < this -> V; i++){
		excessFlow[source] += edges[source][i].first.capacity;
		excessFlow[source] += edges[source][i].second.capacity; // Redundant
	}
}

void Graph::push(int s, int t, int color){
	long long delta;
	if(color == 1){
		delta = min(excessFlow[s], edges[s][t].first.capacity - edges[s][t].first.flow);
		edges[s][t].first.flow += delta;
		edges[t][s].second.flow -= delta;
	}
	else{
		delta = min(excessFlow[s], edges[s][t].second.capacity - edges[s][t].second.flow);
		edges[s][t].second.flow += delta;
		edges[t][s].first.flow -= delta;

		// delta = min(excessFlow[s], edges[t][s].second.capacity - edges[t][s].second.flow);
		// edges[t][s].second.flow += delta;
		// edges[s][t].first.flow -= delta;
	}
	excessFlow[t] += delta;
	excessFlow[s] -= delta;
}

void Graph::relabel(int v){
	long long temp = -1;
	for (int i = 0; i < V; i++){
		if(edges[v][i].first.capacity!=-1){
			if(edges[v][i].first.capacity > edges[v][i].first.flow){
				if(temp == -1 || temp > label[i]){
					temp = label[i];
				}
			}
		}
		if(edges[v][i].second.capacity!=-1){
			if(edges[v][i].second.capacity > edges[v][i].second.flow){
				if(temp == -1 || temp > label[i]){
					temp = label[i];
				}
			}
		}
	}
	label[v] = 1 + temp;
}

long long Graph::maxFlow(int source, int sink){
	long long m;
	long long flow=0;

	active[source] = active[sink] = true;
	initializePreflow(source);

	for (int i = 0; i < V; i++){
		if(edges[source][i].first.capacity!=-1){
			push(source, i, 1);
			if(!active[i]){
				active[i] = true;
				activeNodes.push(i);
			}
		}
		if(edges[source][i].second.capacity!=-1){
			push(source, i, 2);
			if(!active[i]){
				active[i] = true;
				activeNodes.push(i);
			}
		}
	}
	// for (int i= 0; i < V; i++)
	// 	cout <<excessFlow[i]<<'\n';

	int vertexToFix;
	while (!activeNodes.empty()){
		cout<<":"<<activeNodes.size()<<":"<<endl;
		vertexToFix = activeNodes.front();
		active[vertexToFix] = false;
		activeNodes.pop();

		for (int i = 0; i < V && excessFlow[vertexToFix] > 0; i++){
			if(edges[vertexToFix][i].first.capacity!=-1){
				if(edges[vertexToFix][i].first.capacity > edges[vertexToFix][i].first.flow){
					if(label[vertexToFix] > label[i]){
						push(vertexToFix, i, 1);
						if(!active[i] && excessFlow[i] > 0){
							active[i] = true;
							activeNodes.push(i);
						}
					}
					else{
						relabel(vertexToFix);
					}
				}
			}
			if(excessFlow[vertexToFix] <=0 ) break;
			if(edges[vertexToFix][i].second.capacity!=-1){
				if(edges[vertexToFix][i].second.capacity > edges[vertexToFix][i].second.flow){
					if(label[vertexToFix] > label[i]){
						push(vertexToFix, i, 2);
						if(!active[i] && excessFlow[i] > 0){
							active[i] = true;
							activeNodes.push(i);
						}
					}
					else{
						relabel(vertexToFix);
					}
				}
			}
		}
	}
	for(int i=0;i<V;i++){
		if(edges[source][i].first.capacity){
			flow += edges[source][i].first.flow;
			flow += edges[source][i].second.flow;
		}
	}
	return flow;
}

int main()
{
	int i, m, n, x, y;
	long long z;
	cin >> n >> m;
	Graph g(n);
	while (m--)
	{
		cin >> x >> y >> z;
		if(x==y) continue;
		g.addEdge(x - 1, y - 1, z);
		g.addEdge(y - 1, x - 1, z);
	}
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			cout<<"("<<edges[i][j].first.capacity<<","<<edges[i][j].second.capacity<<","<<edges[i][j].first.color<<","<<edges[i][j].second.color<<") ";
		}
		cout<<endl;
	}
	g.initializePreflow(0);
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			cout<<"("<<edges[i][j].first.capacity<<","<<edges[i][j].second.capacity<<","<<edges[i][j].first.color<<","<<edges[i][j].second.color<<") ";
		}
		cout<<endl;
	}
	// cout << g.maxFlow(0,n-1) << endl;
	// int i, m, n, x, y, count = 1;
	// long long z;
 //  	n = -1;
 //  	while (n)
 //  	{
	//     cin >> n;
	//     if (!n)
 //      	break;
 //    	Graph g(n);
 //    	int source, sink;
 //    	cin >> source >> sink >> m;
 //    	// cout << source << ' ' << sink << '\n';
 //    	while (m--)
 //    	{
 //      		cin >> x >> y >> z;
 //      		if (x == y) continue;
 //      		g.addEdge(x - 1, y - 1, z);
 //      		g.addEdge(y - 1, x - 1, z);
 //    	}
 //    	cout << "Network " << count++ << '\n';
 //    	cout << "The bandwidth is " << g.maxFlow(0,n-1) << ".\n\n";
 //  	}
	return 0;
}