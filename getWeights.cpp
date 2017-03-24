#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <IL/il.h>
#include <IL/ilu.h>


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

int cost_function(unsigned char* pixmap, int u, int v){
	// Random cost function for now; change to a sophisticated one later
	int pr, pg, pb, qr, qg, qb, cost;

	pr = pixmap[u*4 + 0];
	pg = pixmap[u*4 + 0];
	pb = pixmap[u*4 + 0];

	qr = pixmap[v*4 + 0];
	qg = pixmap[v*4 + 0];
	qb = pixmap[v*4 + 0];

	cost = (pr-qr)*(pr-qr) + (pg-qg)*(pg-qg) + (pb-qb)*(pb-qb);
	return cost;
}

void createWeights(unsigned char * bitmap, int** weights, int height, int width){
	// V = P U {S(object),T(background)}
	int u, v, B_uv, K = 0, n_nodes;
	n_nodes = (height*width) + 2;
	// {p,q} type edges
	// weights = new int[n_nodes][n_nodes];
	for (int i = 1; i < height - 1; i+=2){
   		for (int j = 1; j < width - 1; j+=2){
   			for(int k = -1; k < 1; k++){
   				for(int l = -1; l < 1; l++){
   					u = i*height + j;
   					v = (i+k)*height + (j+l);
   					B_uv = cost_function(bitmap, u, v);
   					K = std::max(B_uv, K) + 1;
   					weights[u][v] = B_uv;
   				}
   			}
	  	}
	}
	//Assign other weights according to constraints
}


int main(){
	int width, height;
	unsigned char* pixmap;

	ilInit();
	loadImage("./images/large.png", pixmap, width, height);

	return 0;
}
