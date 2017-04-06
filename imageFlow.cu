#include <bits/stdc++.h>
#include <IL/il.h>
#include <IL/ilu.h>

#define BLOCK_SIZE 64
#define HYPERm 2
#define HYPERk 4
#define LAMBDA 2


using namespace std;

// Global temporary storage F
__device__ unsigned long long** F;

unsigned long long B_function(int x, int y){
	return (x - y) * (x - y);
}

unsigned long long R_function(int x, int y){
	//Object 
	if(y == 1){
		return 1;
	}
	return 2;
}

struct Pixel
{
	int pixel_value, hard_constraint, height;
	unsigned long long neighbor_capacities[10]; //Stored in row major form, followed by source and sink
	unsigned long long int  excess;
	bool is_active;
	Pixel()
	{
		this -> hard_constraint = 0;
		this -> height = 0;
		this -> excess = 0;
		this -> is_active = false;
	}
};

void saveImage(const char* filename, int width, int height, unsigned char * bitmap)
{
	ILuint imageID = ilGenImage();
	ilBindImage(imageID);
	ilTexImage(width, height, 0, 1,IL_LUMINANCE, IL_UNSIGNED_BYTE, bitmap);
	iluFlipImage();
	ilEnable(IL_FILE_OVERWRITE);
	ilSave(IL_PNG, filename);
	fprintf(stderr, "Image saved as: %s\n", filename);
}

ILuint loadImage(const char *filename, unsigned char ** bitmap, int &width, int &height)
{
	ILuint imageID = ilGenImage();
	ilBindImage(imageID);
	ILboolean success = ilLoadImage(filename);
	if (!success) return 0;

	width = ilGetInteger(IL_IMAGE_WIDTH);
	height = ilGetInteger(IL_IMAGE_HEIGHT);
	printf("Width: %d\t Height: %d\n", width, height);
	*bitmap = ilGetData();
	return imageID;
}

__global__ void push(Pixel *image_graph, int height, int width)
{
	int offset = height + width;
	int i = (threadIdx.x + blockIdx.x * blockDim.x) + offset;
	int j = (threadIdx.y + blockDim.y * blockIdx.y) + offset;

	int locali = i%BLOCK_SIZE, localj =j%BLOCK_SIZE;

	__shared__ int shared_heights[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
	__shared__ int shared_excess[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

	shared_heights[locali + 1][localj + 1] = image_graph[i * width + j].height;
	shared_excess[locali + 1][localj + 1] = image_graph[i * width + j].excess;

	if(locali == 0){
		shared_excess[0][localj + 1] = image_graph[(i - 1) * width + j].excess;
		shared_heights[0][localj + 1] = image_graph[(i - 1) * width + j].height;
		if(localj == 0){
			shared_excess[0][0] = image_graph[(i - 1) * width + j - 1].excess;
			shared_heights[0][BLOCK_SIZE + 1] = image_graph[(i - 1) * width + j + 1].height;
		}
	}
	else if(localj == BLOCK_SIZE - 1){
		shared_excess[BLOCK_SIZE + 1][localj + 1] = image_graph[(i + 1) * width + j].excess;
		shared_heights[BLOCK_SIZE + 1][localj + 1] = image_graph[(i + 1) * width + j].height;
		if(localj == 0){
			shared_excess[BLOCK_SIZE + 1][0] = image_graph[(i + 1) * width + j - 1].excess;
			shared_heights[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = image_graph[(i + 1) * width + j + 1].height;
		}	
	}
	else if(localj == 0){
		shared_excess[locali + 1][0] = image_graph[i * width + j - 1].excess;
		shared_heights[locali + 1][0] = image_graph[i * width + j - 1].height;
	}
	else if(localj == BLOCK_SIZE - 1){
		shared_excess[locali + 1][BLOCK_SIZE + 1] = image_graph[i * width + j + 1].excess;
		shared_heights[locali + 1][BLOCK_SIZE + 1] = image_graph[i * width + j + 1].height;
	}
	__syncthreads();

	// Row major traversal of neighbors of a pixel (i,j)
	int x_offsets[] = {-1,-1,-1,0,0,1,1,1};
	int y_offsets[] = {-1,0,1,-1,1,-1,0,1};

	int dest_x, dest_y;
	// Check spatial neighbors
	for(int i=0;i<8;i++){
		dest_x = locali + x_offsets[i] + 1;
		dest_y = localj + y_offsets[i] + 1;
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
		dest_x = i + x_offsets[k] + 1;
		dest_y = j + y_offsets[k] + 1;
		aggregate_flow += F[dest_x][dest_y][7 - k];
	}

	// Run same condition as above for source, sink

	image_graph[i * width + j].excess += aggregate_flow;
}

__global__ void localRelabel(Pixel *image_graph, int height, int width)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int locali = i%BLOCK_SIZE, localj =j%BLOCK_SIZE;

	__shared__ int shared_heights[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
	__shared__ bool shared_flags[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

	shared_heights[locali + 1][localj + 1] = image_graph[i * width + j].height;
	shared_flags[locali + 1][localj + 1] = image_graph[i * width + j].is_active;

	__syncthreads();

	// Row major traversal of neighbors of a pixel (i,j)
	int x_offsets[] = {-1,-1,-1,0,0,1,1,1};
	int y_offsets[] = {-1,0,1,-1,1,-1,0,1};

	int dest_x, dest_y;
	unsigned long long aggregate_flow = 0;
	int min_height = INT_MAX;

	// Check spatial neighbors
	for(int i=0;i<8;i++){
		dest_x = locali + x_offsets[i] + 1;
		dest_y = localj + y_offsets[i] + 1;
		if( image_graph[dest_x * width + dest_y].is_active){
			min_height = min(min_height, shared_heights[dest_x][dest_y]);
		}
	}

	// Run same condition as above for source, sink

	image_graph[i * width + j].height = min_height;
}

__global__ void globalRelabel(Pixel *image_graph, int height, int width, int iteration)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int locali = i%BLOCK_SIZE, localj =j%BLOCK_SIZE;

	//No divergence
	if(iteration == 1){
		for (int l = 0; l < 8; l++)
			if(image_graph[i * width + j].neighbor_capacities[l] > image_graph[i * width + j].excess){
				image_graph[i * width + j].height = 1;
		}
	}
	else{
		__shared__ int shared_heights[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
		
		shared_heights[locali + 1][localj + 1] = image_graph[i * width + j].height;
		__syncthreads();

		bool satisfied = false;
		int dest_x, dest_y;

		// Row major traversal of neighbors of a pixel (i,j)
		int x_offsets[] = {-1,-1,-1,0,0,1,1,1};
		int y_offsets[] = {-1,0,1,-1,1,-1,0,1};

		for(int i1=0; i1<8; i1++){
			dest_x = locali + x_offsets[i1] + 1;
			dest_y = localj + y_offsets[i1] + 1;
			if(shared_heights[dest_x][dest_y] == iteration){
				satisfied = true;
				break;
			}
		}

		if(satisfied){
			shared_heights[locali + 1][localj + 1] = iteration + 1;
			image_graph[i * width + j].height = iteration + 1;
		}
	}
}

__global__ void initNeighbors(Pixel *imagegraph, int height, int width, unsigned long long* K){

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	int locali = i%BLOCK_SIZE, localj =j%BLOCK_SIZE;

	__shared__ Pixel block_pixels[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
	block_pixels[locali + 1][localj + 1] = imagegraph[i * width + j];

	// Load pixels from boundary neighbors


	__syncthreads();

	// Row major traversal of neighbors of a pixel (i,j)
	int x_offsets[] = {-1,-1,-1,0,0,1,1,1};
	int y_offsets[] = {-1,0,1,-1,1,-1,0,1};

	unsigned long long max_k = 0;
	int dest_x, dest_y;

	for(int i=0; i<8; i++){
		dest_x = locali + x_offsets[i] + 1;
		dest_y = localj + y_offsets[i] + 1;
		block_pixels[locali + 1][localj + 1].neighbor_capacities[i] = B_function(block_pixels[locali + 1][localj + 1].pixel_value, block_pixels[dest_x][dest_y].pixel_value);
		max_k += block_pixels[locali + 1][localj + 1].neighbor_capacities[i];
	}
	max_k++;

	__shared__ unsigned long long blockmax;

	if(threadIdx.x == 0 && threadIdx.y == 0){
		blockmax = INT_MAX;
	}
	__syncthreads();

	atomicMax(&blockmax, max_k);
	__syncthreads();

	if(threadIdx.x == 0 && threadIdx.y == 0){
		atomicMax(K, blockmax);
	}

	// Won't work; copy neighbor wise
	imagegraph[i * width + j] = block_pixels[locali + 1][localj + 1];
}

//Also accept hard and soft constraints array
__global__ void initConstraints(Pixel *imagegraph, int height, int width, unsigned long long K){

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	// {p,S} edge
	imagegraph[i * width + j].neighbor_capacities[8] = (imagegraph[i* width + j].hard_constraint[i][j] == 0) * K
										+ (imagegraph[i* width + j].hard_constraint[i][j] == 1) * LAMBDA * R_function(imagegraph[i* width + j].pixel_value, -1);

	// {p,T} edge
	imagegraph[i* width + j].neighbor_capacities[9] = (imagegraph[i* width + j].hard_constraint[i][j] == -1) * K
										+ (imagegraph[i* width + j].hard_constraint[i][j] == 0) * LAMBDA * R_function(imagegraph[i* width + j].pixel_value, 1);
}


int main(int argc, char* argv[])
{
	int width, height;
	unsigned long long* K = new unsigned long long;
	*K = LLONG_MAX;
	bool* convergence_flag = new bool;

	unsigned char *image;
	Pixel *image_graph, *cuda_image_graph;
	
	ilInit();

	ILuint image_id = loadImage(argv[1], &image, width, height);
	int pixel_memsize = (width+1) * (height+1) * sizeof(Pixel);
	if(image_id == 0) {fprintf(stderr, "Error while reading image... aborting.\n"); exit(0);}

	//Pixel graph with padding to avoid convergence in kernels for boundary pixels
	image_graph = (Pixel*)malloc(pixel_memsize); 

	assert(cudaSuccess == cudaMalloc((void**) &cuda_image_graph, pixel_memsize));
	assert(cudaSuccess == cudaMemcpy(cuda_image_graph, image_graph, pixel_memsize, cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks(width/BLOCK_SIZE, height/BLOCK_SIZE);
	
	// Load weights in graph using kernel call/host loops
	initNeighbors<<<numBlocks, threadsPerBlock>>>(cuda_image_graph, height, width, K);
	initConstraints<<<numBlocks, threadsPerBlock>>>(cuda_image_graph, height, width, *K);

	int iteration = 1;
	while(!convergence_flag){
		for(int i=0; i<HYPERk; i++){
			for(int j=0; j<HYPERm; j++){
				push<<<numBlocks, threadsPerBlock>>>(cuda_image_graph, height, width);
				pull<<<numBlocks, threadsPerBlock>>>(cuda_image_graph, width, height);
			}
			localRelabel<<<numBlocks, threadsPerBlock>>>(cuda_image_graph, width, height);
		}
		globalRelabel<<<numBlocks, threadsPerBlock>>>(cuda_image_graph, width, height, iteration);
		iteration++;
	}

	// Load segmented image from graph using another kernel and display it

	return 0;	
}
