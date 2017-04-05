#include <bits/stdc++.h>

#define FLATIMAGESIZE 316 * 300
#define N 256
#define M 256
#define BLOCK_SIZE 64

using namespace std;

struct Pixel
{
	unsigned long long neighbor_capacities[10]; //Stored in row major form, followed by source and sink
	unsigned long long int flow, height, excess;
	bool isActive;
	Pixel()
	{
		this -> isActive = false;
	}
};


__global__ void push(Pixel *image_graph, int height, int width)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	int locali = i%BLOCK_SIZE, localj =j%BLOCK_SIZE;

	__shared__ int shared_heights[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int shared_excess[BLOCK_SIZE][BLOCK_SIZE];

	shared_heights[locali][localj] = image_graph[i][j].height;
	shared_excess[locali][localj] = image_graph[i][j].excess;
	__syncthreads();

	bool is_active = false;

	// Row major traversal of neighbors of a pixel (i,j)
	int x_offsets[] = {-1,-1,-1,0,0,1,1,1};
	int y_offsets[] = {-1,0,1,-1,1,-1,0,1};

	int dest_x, dest_y;
	// Check spatial neighbors
	for(int i=0;i<8;i++){
		dest_x = locali+x_offsets[i];
		dest_y = localj+y_offsets[i];
		if(shared_heights[dest_x][dest_y] + 1 == shared_heights[localj][localj]){
			shared_excess[dest_x][dest_y] += shared_excess[locali][localj]; //push e(u) to eligible neighbors
		}
	}
	// Run same condition as above for source, sink

	__syncthreads();
	//store excess flow in a global 'F' array
}


__global__ void pull(Pixel *image_graph, int height, int width)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	// Row major traversal of neighbors of a pixel (i,j)
	int x_offsets[] = {-1,-1,-1,0,0,1,1,1};
	int y_offsets[] = {-1,0,1,-1,1,-1,0,1};

	int dest_x, dest_y;
	unsigned long long aggregate_flow = 0;
	// Check spatial neighbors
	for(int k=0;k<8;k++){
		dest_x = i+x_offsets[k];
		dest_y = j+y_offsets[k];
		aggregate_flow += F[dest_x][dest_y].weights[7 - k];
	}

	// Run same condition as above for source, sink

	image_graph[i][j].excess += aggregate_flow;
}

__global__ void localRelabel(Pixel *image_graph, int height, int width)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int locali = i%BLOCK_SIZE, localj =j%BLOCK_SIZE;

	__shared__ int shared_heights[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ bool shared_flags[BLOCK_SIZE][BLOCK_SIZE];

	shared_heights[locali][localj] = image_graph[i][j].height;
	shared_flags[locali][localj] = image_graph[i][j].is_active;

	__syncthreads();

	// Row major traversal of neighbors of a pixel (i,j)
	int x_offsets[] = {-1,-1,-1,0,0,1,1,1};
	int y_offsets[] = {-1,0,1,-1,1,-1,0,1};

	int dest_x, dest_y;
	unsigned long long aggregate_flow = 0;
	int min_height = INT_MAX;

	// Check spatial neighbors
	for(int i=0;i<8;i++){
		dest_x = locali+x_offsets[i];
		dest_y = localj+y_offsets[i];
		if( is active or passive node){
			min_height = min(min_height, shared_heights[dest_x][dest_y]);
		}
	}

	// Run same condition as above for source, sink

	image_graph[i][i].height = min_height;
}

__global__ void globalRelabel(Pixel *image_graph, int height, int width)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int locali = i%BLOCK_SIZE, localj =j%BLOCK_SIZE;

	//run BFS
}


int main()
{
	return 0;	
}
