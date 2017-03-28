#include <bits/stdc++.h>
#include <IL/il.h>
#include <IL/ilu.h>

#define LAMBDA 2

using namespace std;

void loadImage(const char* filename, unsigned char* pixmap, int &width, int &height){
	ILuint ImgId = 0;
	ilGenImages(1, &ImgId);
	ilBindImage(ImgId);
	ilLoadImage(filename);

	width = ilGetInteger(IL_IMAGE_WIDTH);
	height = ilGetInteger(IL_IMAGE_HEIGHT);
	pixmap = new unsigned char[width * height * 3];
	ilCopyPixels(0, 0, 0, width, height, 1, IL_RGB, 
	IL_UNSIGNED_BYTE, pixmap);

	ilBindImage(0);
	ilDeleteImage(ImgId);

	printf( "Height:%d Width:%d\n", height, width);
}

int background_function(unsigned char* pixmap, int u, int v){
	// Random cost function for now; change to a sophisticated one later
	return 0;
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
	return 1;
}

void createWeights(unsigned char * pixmap, int** weights, int height, int width){
	// V = P U {S(object),T(background)}
	set< pair<int,int> > hard_object, hard_background;
	bool foreground_map[height][width];
	for (int i = 0; i < height ; i++){
		for (int j = 0; j < width ; j++){
			foreground_map[i][j] = false;
		}
	}
	int u, v, B_uv, K = 0, n_nodes;
	n_nodes = (height*width) + 2;
	// {p,q} type edges
	for (int i = 1; i < height-1 ; i+=2){
		for (int j = 1; j < width-1 ; j+=2){
			for(int k = -1; k < 1; k++){
				for(int l = -1; l < 1; l++){
					u = i*width + j;
					v = (i+k)*width + (j+l);
					B_uv = background_function(pixmap, u, v);
					K = std::max(B_uv, K) + 1;
					weights[u][v] = B_uv;
					weights[v][u] = B_uv;
				}
			}
	  	}
	}
	// {p,S} type edges
	for (int i = 1; i < height ; i+=1){
		for (int j = 1; j < width ; j+=1){
			u = i*width + j;
			if(hard_object.find(make_pair(u,0)) != hard_object.end()){
				weights[u][0] = weights[0][u] = K;
			}
			else if(hard_background.find(make_pair(u,0)) != hard_background.end()){
				weights[u][0] = weights[0][u] = 0;	
			}
			else{
				weights[u][0] = LAMBDA * region_function(pixmap, u, 0, false);
				weights[0][u] = weights[u][0];
			}
		}
	}
	// {p,T} type edges
	for (int i = 1; i < height ; i+=1){
		for (int j = 1; j < width ; j+=1){
			u = i*width + j;
			v = n_nodes-1;
			if(hard_object.find(make_pair(u,v)) != hard_object.end()){
				weights[u][v] = weights[v][u] = 0;
			}
			else if(hard_background.find(make_pair(u,v)) != hard_background.end()){
				weights[u][v] = weights[v][u] = K;	
			}
			else{
				weights[u][v] = LAMBDA * region_function(pixmap, u, v, true);
				weights[v][u] = weights[u][v];
			}
		}
	}
	printf("Woohoo\n");
}


int main(){
	int width, height;
	unsigned char* pixmap;

	ilInit();
	loadImage("./images/large.jpg", pixmap, width, height);

	int n_nodes = (width*height) + 2;
	int **weights = new int*[n_nodes];
	for (int i = 0; i < n_nodes ; i+=1){
		weights[i] = new int[n_nodes];
	}

	createWeights(pixmap, weights, height, width);

	for(int i=0;i<n_nodes;i++){
		for(int j=0;j<n_nodes; j++){
			printf("%d %d %d\n", i, j, weights[i][j]);
		}
	}
	return 0;
}
