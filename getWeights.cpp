#include <bits/stdc++.h>
#include <IL/il.h>
#include <IL/ilu.h>

#define LAMBDA 2

using namespace std;


class Graph
{
public:
	int V;
	list<int> *adj;
	map <int, int> *edges;
	Graph(int V);
	void addEdge(int u, int v, long long capacity);
};

Graph::Graph(int V)
{
	this -> V = V;
	edges = new map<int, int>[V];
	adj = new list<int> [V];
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

void loadImage(const char* filename, unsigned char* pixmap, int &width, int &height){
	ILuint ImgId = 0;
	ilGenImages(1, &ImgId);
	ilBindImage(ImgId);
	ilLoadImage(filename);

	width = ilGetInteger(IL_IMAGE_WIDTH);
	height = ilGetInteger(IL_IMAGE_HEIGHT);
	pixmap = new unsigned char[width * height * 3];
	ilCopyPixels(0, 0, 0, width, height, 1, IL_RGB, IL_UNSIGNED_BYTE, pixmap);

	ilBindImage(0);
	ilDeleteImage(ImgId);
}

int background_function(unsigned char* pixmap, int u, int v){
	// Random cost function for now; change to a sophisticated one later
	return 2;
	return (pixmap[u] - pixmap[v]) * (pixmap[u] - pixmap[v]);
	int pr, pg, pb, qr, qg, qb, cost;

	pr = pixmap[u*4 + 0];
	pg = pixmap[u*4 + 1];
	pb = pixmap[u*4 + 2];

	qr = pixmap[v*4 + 0];
	qg = pixmap[v*4 + 1];
	qb = pixmap[v*4 + 2];

	cost = (pr-qr)*(pr-qr) + (pg-qg)*(pg-qg) + (pb-qb)*(pb-qb);
	return cost;
}


int region_function(unsigned char* pixmap, int u, int v, bool is_foreground){
	// Random cost function for now; change to a sophisticated one later
	return 1*is_foreground;
	return (u-v)*(u-v)*is_foreground;
}

void createWeights(unsigned char * pixmap, int** weights, int height, int width, 
	set< pair<int,int> > hard_object, set< pair<int,int> > hard_background){
	// V = P U {S(object),T(background)}
	int u, v, B_uv, K = 0, n_nodes;
	n_nodes = (height*width) + 2;
	// {p,q} type edges
	for (int i = 1; i < height-1 ; i+=2){
		for (int j = 1; j < width-1 ; j+=2){
			for(int k = -1; k < 1; k++){
				for(int l = -1; l < 1; l++){
					u = i*width + j;
					v = (i+k)*width + (j+l);
					if(k || l){
						B_uv = background_function(pixmap, u, v);
						K = std::max(B_uv, K) + 1;
						weights[u+1][v+1] = weights[v+1][u+1] = B_uv;
					}
				}
			}
	  	}
	}
	// {p,S} type edges
	for (int i = 0; i < height ; i+=1){
		for (int j = 0; j < width ; j+=1){
			u = i*width + j;
			if(hard_object.find(make_pair(u,0)) != hard_object.end()){
				weights[u+1][0] = weights[0][u+1] = K;
			}
			else if(hard_background.find(make_pair(u,0)) != hard_background.end()){
				weights[u+1][0] = weights[0][u+1] = 0;	
			}
			else{
				weights[u+1][0] = LAMBDA * region_function(pixmap, u, 0, false);
				weights[0][u+1] = weights[u+1][0];
			}
		}
	}
	// {p,T} type edges
	for (int i = 0; i < height ; i+=1){
		for (int j = 0; j < width ; j+=1){
			u = i*width + j;
			v = n_nodes-1;
			if(hard_object.find(make_pair(u,v)) != hard_object.end()){
				weights[u+1][v] = weights[v][u+1] = 0;
			}
			else if(hard_background.find(make_pair(u,v)) != hard_background.end()){
				weights[u+1][v] = weights[v][u+1] = K;	
			}
			else{
				weights[u+1][v] = LAMBDA * region_function(pixmap, u, v, true);
				weights[v][u+1] = weights[u+1][v];
			}
		}
	}
}

void createWeights2(unsigned char * pixmap, Graph g, int height, int width, 
	set< pair<int,int> > hard_object, set< pair<int,int> > hard_background){
	// V = P U {S(object),T(background)}
	int u, v, B_uv, K = 0, n_nodes;
	n_nodes = (height*width) + 2;
	// {p,q} type edges
	for (int i = 1; i < height-1 ; i+=2){
		for (int j = 1; j < width-1 ; j+=2){
			for(int k = -1; k < 1; k++){
				for(int l = -1; l < 1; l++){
					u = i*width + j;
					v = (i+k)*width + (j+l);
					if(k || l){
						B_uv = background_function(pixmap, u, v);
						K = std::max(B_uv, K) + 1;
						g.addEdge(u + 1, v + 1, B_uv);
						g.addEdge(v + 1, u + 1, B_uv);
					}
				}
			}
	  	}
	}
	// {p,S} type edges
	for (int i = 0; i < height ; i+=1){
		for (int j = 0; j < width ; j+=1){
			u = i*width + j;
			if(hard_object.find(make_pair(u,0)) != hard_object.end()){
				g.addEdge(u + 1, 0, K);
				g.addEdge(0, u + 1, K);
			}
			else if(hard_background.find(make_pair(u,0)) != hard_background.end()){
				g.addEdge(u + 1, 0, 0);
				g.addEdge(0, u + 1, 0);	
			}
			else{
				g.addEdge(u + 1, 0, LAMBDA * region_function(pixmap, u, 0, false));
				g.addEdge(0, u + 1, LAMBDA * region_function(pixmap, u, 0, false));
			}
		}
	}
	// {p,T} type edges
	for (int i = 0; i < height ; i+=1){
		for (int j = 0; j < width ; j+=1){
			u = i*width + j;
			v = n_nodes-1;
			if(hard_object.find(make_pair(u,v)) != hard_object.end()){
				g.addEdge(u + 1, v, 0);
				g.addEdge(v, u + 1, 0);
			}
			else if(hard_background.find(make_pair(u,v)) != hard_background.end()){
				g.addEdge(u + 1, v, K);
				g.addEdge(v, u + 1, K);
			}
			else{
				g.addEdge(u + 1, v, LAMBDA * region_function(pixmap, u, v, true));
				g.addEdge(v, u + 1, LAMBDA * region_function(pixmap, u, v, true));
			}
		}
	}
}

void load_constraints(const char* filename, set<pair<int,int> > points){
	ifstream input_stream(filename);
	if (!input_stream) cerr << "Can't open input file!\n";
	// one line
	string line;
	int x,y;
	// extract all the text from the input file
	while (getline(input_stream, line)) {
		istringstream iss(line);
		string sub;
	   	iss >> sub;
	   	x = stoi(sub);
		iss >> sub;
		y = stoi(sub);
		points.insert(make_pair(x,y));
	}
}

int main( int argc, char *argv[]){
	int width, height;
	unsigned char* pixmap;

	ilInit();
	loadImage(argv[1], pixmap, width, height);

	int n_nodes = (width*height) + 2;
	int **weights = new int*[n_nodes];
	for (int i = 0; i < n_nodes ; i+=1){
		weights[i] = new int[n_nodes];
	}

	Graph weight_graph(n_nodes);

	set<pair<int,int> > object, background;
	load_constraints(argv[2], object);
	load_constraints(argv[3], background);

	// createWeights(pixmap, weights, height, width, object, background);
	createWeights2(pixmap, weight_graph, height, width, object, background);

	// int n_edges=0;
	// for(int i=0;i<n_nodes;i++){
	// 	for(int j=0;j<n_nodes;j++){
	// 		if(weights[i][j]>0){
	// 			n_edges++;
	// 		}
	// 	}
	// }
	// cout<<n_nodes<<" "<<n_edges<<"\n";
	// for(int i=0;i<n_nodes;i++){
	// 	for(int j=0;j<n_nodes;j++){
	// 		if(weights[i][j]>0)
	// 		cout<<i+1<<" "<<j+1<<" "<<weights[i][j]<<"\n";
	// 	}
	// }

	int n_edges=0;
	for(int i=0;i<n_nodes;i++){
		for (iter = G.edges[i].begin(); iter != G.edges[i].end(); iter++)
		{
			y = *iter;
			if (!reachable[y] && edges[x][y] > 0)
			{
				reachable[y] = true;
				reachableVertices.push(y);
			}
		}
		for(int j=0;j<n_nodes;j++){
			if(weights[i][j]>0){
				n_edges++;
			}
		}
	}
	cout<<n_nodes<<" "<<n_edges<<"\n";
	for(int i=0;i<n_nodes;i++){
		for(int j=0;j<n_nodes;j++){
			if(weights[i][j]>0)
			cout<<i+1<<" "<<j+1<<" "<<weights[i][j]<<"\n";
		}
	}

	return 0;
}
