#include <bits/stdc++.h>
#include <IL/il.h>
#include <IL/ilu.h>

#define BLOCK_SIZE 32
#define HYPERm 2
#define HYPERk 4
#define LAMBDA 2

using namespace std;

__device__ unsigned long long B_function(int x, int y){
	// return (x - y) * (x - y);
	return abs(x - y);
}

__device__ unsigned long long R_function(int x, int y){
	//Object 
	if(y == 1){
		return 1;
	}
	return 2;
}

struct Pixel{
	int pixel_value, hard_constraint, height;
	unsigned long long neighbor_capacities[10]; //Stored in row major form, followed by source and sink
	unsigned long long neighbor_flows[10];
	unsigned long long int excess;
	bool is_active;
	Pixel(){
		this -> hard_constraint = 0;
		this -> height = 0;
		this -> excess = 0;
		this -> is_active = false;
	}
};

struct Terminal{
	unsigned long long int excess;
	bool is_active;
	int height;
	Terminal(){
		this -> is_active = false;
		this -> height = 0;
		this -> excess = 0;
	}
};

void saveImage(const char* filename, int width, int height, unsigned char * bitmap){
	ILuint imageID = ilGenImage();
	ilBindImage(imageID);
	ilTexImage(width, height, 0, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, bitmap);
	iluFlipImage();
	ilEnable(IL_FILE_OVERWRITE);
	ilSave(IL_PNG, filename);
	fprintf(stderr, "Image saved as: %s\n", filename);
}

ILuint loadImage(const char *filename, unsigned char ** bitmap, int &width, int &height){
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

__global__ void push(Pixel *image_graph, unsigned long long *F, Terminal *source, Terminal *sink, int height, int width, int *convergence_flag){
	int i = (threadIdx.x + blockIdx.x * blockDim.x) + 1;
	int j = (threadIdx.y + blockDim.y * blockIdx.y) + 1;

	if (i <= height && j <= width){

		// unsigned long long *neighbor_flows = image_graph[i * width + j].neighbor_flows;
		// unsigned long long *neighbor_capacities = image_graph[i * width + j].neighbor_capacities;
		unsigned long long excess = image_graph[i * width + j].excess;

		// Row major traversal of neighbors of a pixel (i,j)
		int x_offsets[] = {-1, -1, -1, 0, 0, 1, 1, 1};
		int y_offsets[] = {-1, 0, 1, -1, 1, -1, 0, 1};

		int thread_flag = 0;
		int dest_x, dest_y;


		// Check spatial neighbors
		for(int l = 0; l < 8; l++){
			dest_x = i + x_offsets[l];
			dest_y = j + y_offsets[l];
			if(image_graph[dest_x * width + dest_y].height + 1 == image_graph[i * width + j].height){
				int flow = min(image_graph[i * width + j].neighbor_capacities[l] - image_graph[i * width + j].neighbor_flows[l], excess);
				atomicAdd(&(image_graph[i * width + j].excess) , -flow) ;
				atomicAdd(&(image_graph[dest_x * width + dest_y].excess), flow) ;
				atomicAdd(&(image_graph[i * width + j].neighbor_capacities[l]) , -flow) ;
				atomicAdd(&(image_graph[dest_x * width + dest_y].neighbor_capacities[7 - l]), flow) ;
				thread_flag = 1;
			}
		}

		unsigned long long flow = min(image_graph[i * width + j].excess, image_graph[i * width + j].neighbor_capacities[9]);
		atomicAdd(&image_graph[i * width + j].neighbor_flows[9], flow);
		if (image_graph[i * width + j].excess == flow)
			atomicAdd(&image_graph[i * width + j].excess, -flow);
		atomicAdd(&(sink -> excess), flow);
		__syncthreads();
		// Update flags
		atomicOr(convergence_flag, thread_flag);		
		// printf("%d ", *convergence_flag);
	}
}

// __global__ void pull(Pixel *image_graph, unsigned long long *F, Terminal *source, Terminal *sink, int height, int width){
// 	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
// 	int j = threadIdx.y + blockDim.y * blockIdx.y + 1;

// 	// Should be <=, but fails for that
// 	if (i < height && j < width){
// 		unsigned long long aggregate_flow = 0;
// 		// Row major traversal of neighbors of a pixel (i,j)
// 		int x_offsets[] = {-1, -1, -1, 0, 0, 1, 1, 1};
// 		int y_offsets[] = {-1, 0, 1, -1, 1, -1, 0, 1};

// 		int dest_x, dest_y;

// 		// Check spatial neighbors
// 		for(int k = 0; k < 8; k++){
// 			dest_x = i + x_offsets[k];
// 			dest_y = j + y_offsets[k];
// 			aggregate_flow += F[dest_x * width + dest_y];
// 		}

// 		aggregate_flow += source->excess;
// 		// aggregate_flow += source->excess;

// 		image_graph[i * width + j].excess += aggregate_flow;
// 	}
// }

// __global__ void localRelabel(Pixel *image_graph, Terminal *source, Terminal *sink, int height, int width){
// 	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
// 	int j = threadIdx.y + blockDim.y * blockIdx.y + 1;
// 	int locali = (i - 1) % BLOCK_SIZE, localj = (j - 1) % BLOCK_SIZE;

// 	if (i <= height && j <= width){
// 		__shared__ int shared_heights[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
// 		// __shared__ bool shared_flags[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

// 		shared_heights[locali + 1][localj + 1] = image_graph[i * width + j].height;
// 		// shared_flags[locali + 1][localj + 1] = image_graph[i * width + j].is_active;

// 		//Boundary pixels of grid
// 		if(locali == 0){
// 			shared_heights[0][localj + 1] = image_graph[(i - 1) * width + j].height;
// 			if(localj == 0){
// 				shared_heights[0][0] = image_graph[(i - 1) * width + (j - 1)].height;
// 			}
// 			else if(localj == BLOCK_SIZE - 1){
// 				shared_heights[0][BLOCK_SIZE + 1] = image_graph[(i - 1) * width + (j + 1)].height;
// 			}
// 		}
// 		else if(locali == BLOCK_SIZE - 1){
// 			shared_heights[BLOCK_SIZE + 1][localj + 1] = image_graph[(i + 1) * width + j].height;
// 			if(localj == 0){
// 				shared_heights[BLOCK_SIZE + 1][0] = image_graph[(i + 1) * width + (j - 1)].height;
// 			}
// 			else if(localj == BLOCK_SIZE - 1){
// 				shared_heights[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = image_graph[(i + 1) * width + (j + 1)].height;
// 			}
// 		}
// 		else if(localj == 0){
// 			shared_heights[locali + 1][0] = image_graph[i * width + (j - 1)].height;
// 		}
// 		else if(localj == BLOCK_SIZE - 1){
// 			shared_heights[locali + 1][BLOCK_SIZE + 1] = image_graph[i * width + (j + 1)].height;
// 		}

// 		__syncthreads();

// 		// Row major traversal of neighbors of a pixel (i,j)
// 		int x_offsets[] = {-1, -1, -1, 0, 0, 1, 1, 1};
// 		int y_offsets[] = {-1, 0, 1, -1, 1, -1, 0, 1};

// 		int dest_x, dest_y;
// 		int min_height = INT_MAX;

// 		// Check spatial neighbors
// 		for(int l = 0; l < 8; l++){
// 			dest_x = (locali + 1) + x_offsets[l];
// 			dest_y = (localj + 1) + y_offsets[l];
// 			// if(image_graph[dest_x * width + dest_y].excess > 0 && image_graph[dest_x * width + dest_y].excess != image_graph[dest_x * width + dest_y].){
// 			// 	min_height = min(min_height, shared_heights[dest_x][dest_y]);
// 			// }
// 		}

// 		// if(source->is_active){
// 		// 		min_height = min(min_height, source->height);
// 		// 	}

// 		// if(sink->is_active){
// 		// 	min_height = min(min_height, sink->height);
// 		// }

// 		image_graph[i * width + j].height = min_height + 1;
// 	}
// }

__global__ void localRelabel(Pixel *image_graph, int height, int width)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.x * blockDim.y + 1;

	if (i <= height && j <= width)
	{
		// Row major traversal of neighbors of a pixel (i,j)
		int x_offsets[] = {-1, -1, -1, 0, 0, 1, 1, 1};
		int y_offsets[] = {-1, 0, 1, -1, 1, -1, 0, 1};

		int dest_x, dest_y, min_height = image_graph[i * width + j].height;
		for(int l = 0; l < 8; l++){
			dest_x = i + x_offsets[l];
			dest_y = j + y_offsets[l];
			min_height = min(min_height, image_graph[dest_x * width + dest_y].height);
		}
		image_graph[i * width + j].height = min(min_height + 1, image_graph[i * width + j].height);
	}
}

// __global__ void globalRelabel(Pixel *image_graph, int height, int width, int iteration){
// 	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
// 	int j = threadIdx.y + blockDim.y * blockIdx.y + 1;

// 	if (i <= height && j <= width){

// 		//No divergence
// 		if(iteration == 1){
// 			for (int l = 0; l < 8; l++)
// 				if(image_graph[i * width + j].neighbor_capacities[l] > image_graph[i * width + j].excess){
// 					image_graph[i * width + j].height = 1;
// 			}
// 		}
// 		else{

// 			bool satisfied = false;
// 			int dest_x, dest_y;

// 			// Row major traversal of neighbors of a pixel (i,j)
// 			int x_offsets[] = {-1, -1, -1, 0, 0, 1, 1, 1};
// 			int y_offsets[] = {-1, 0, 1, -1, 1, -1, 0, 1};

// 			for(int l = 0; l < 8; l++){
// 				dest_x = (locali + 1) + x_offsets[l];
// 				dest_y = (localj + 1) + y_offsets[l];
// 				if(shared_heights[dest_x][dest_y] == iteration){
// 					satisfied = true;
// 					break;
// 				}
// 			}

// 			if(satisfied){
// 				shared_heights[locali + 1][localj + 1] = iteration + 1;
// 				image_graph[i * width + j].height = iteration + 1;
// 			}
// 		}
// 	}
// }

__global__ void initNeighbors(Pixel *image_graph, unsigned char* raw_image, int height, int width, unsigned long long int* K)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockDim.y * blockIdx.y + 1;

	if (i <= height && j <= width){
		image_graph[i * width + j].pixel_value = raw_image[(i - 1) * width + j - 1];

		// Row major traversal of neighbors of a pixel (i,j)
		int x_offsets[] = {-1, -1, -1, 0, 0, 1, 1, 1};
		int y_offsets[] = {-1, 0, 1, -1, 1, -1, 0, 1};

		unsigned long long int max_k = 0;
		unsigned long long edge_weight = 0;
		int dest_x, dest_y;

		for(int k = 0; k < 8; k++){
			dest_x = i + x_offsets[k];
			dest_y = j + y_offsets[k];
			edge_weight = B_function(image_graph[i * width + j].pixel_value, image_graph[dest_x * width + dest_y].pixel_value	);
			image_graph[i * width + j].neighbor_capacities[k] = edge_weight;
			image_graph[i * width + j].neighbor_flows[k] = 0;
			max_k += edge_weight;
		}
		max_k++;
		__syncthreads();
		
		atomicMax(K, max_k);
	}
}

//Also accept hard and soft constraints array
// __global__ void initConstraints(Pixel *image_graph, int height, int width, unsigned long long K){

// 	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
// 	int j = threadIdx.y + blockDim.y * blockIdx.y + 1;

// 	if (i <= height && j <= height){
// 		// {p,S} edge
// 		image_graph[i * width + j].neighbor_capacities[8] = (image_graph[i * width + j].hard_constraint == 0) * K
// 										+ (image_graph[i * width + j].hard_constraint == 1) * LAMBDA * R_function(image_graph[i * width + j].pixel_value, -1);

// 		// {p,T} edge
// 		image_graph[i * width + j].neighbor_capacities[9] = (image_graph[i * width + j].hard_constraint == -1) * K
// 										+ (image_graph[i * width + j].hard_constraint == 0) * LAMBDA * R_function(image_graph[i * width + j].pixel_value, 1);
// 	}
// }


int main(int argc, char* argv[]){
	int width, height;
	unsigned long long* K = new unsigned long long;
	*K = LLONG_MAX;
	int* convergence_flag = new int, *convergence_flag_gpu;
	*convergence_flag = 0;

	unsigned char *image, *cuda_image;
	unsigned long long *K_gpu, *F_gpu;
	Pixel *image_graph, *cuda_image_graph;
	Terminal *source, *sink, *cuda_source, *cuda_sink;
	
	ilInit();

	ILuint image_id = loadImage(argv[1], &image, width, height);
	int pixel_memsize = (width + 1) * (height + 1) * sizeof(Pixel);
	if(image_id == 0) {fprintf(stderr, "Error while reading image... aborting.\n"); exit(0);}

	//Pixel graph with padding to avoid divergence in kernels for boundary pixels
	image_graph = (Pixel*)malloc(pixel_memsize);
	source = new Terminal;
	sink = new Terminal;	

	cudaMalloc((void**)&F_gpu, (width + 1) * (height + 1) * sizeof(unsigned long long));
	cudaMalloc((void**)&convergence_flag_gpu, sizeof(int));
	cudaMalloc((void**)&cuda_image_graph, pixel_memsize);
	cudaMalloc((void**)&cuda_image, width * height * sizeof(unsigned char));
	cudaMalloc((void**)&K_gpu, sizeof(unsigned long long));
	cudaMalloc((void**)&cuda_source, sizeof(Terminal));
	cudaMalloc((void**)&cuda_sink, sizeof(Terminal));
	
	//Set properties of source and sink nodes

	cudaMemcpy(cuda_image_graph, image_graph, pixel_memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_image, image, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(K_gpu, K, sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(convergence_flag_gpu, convergence_flag, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_source, source, sizeof(Terminal), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_sink, sink, sizeof(Terminal), cudaMemcpyHostToDevice);


	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks(height / BLOCK_SIZE + 1, width / BLOCK_SIZE + 1);

	// Load weights in graph using kernel call/host loops
	initNeighbors<<<numBlocks, threadsPerBlock>>>(cuda_image_graph, cuda_image, height, width, K_gpu);
	assert(cudaSuccess == cudaGetLastError());
	printf("Initialized spatial weight values\n");

	// cudaMemcpy(image_graph, cuda_image_graph, pixel_memsize, cudaMemcpyDeviceToHost);
	// for (int i = 0; i < (width + 1) * (height + 1); i++)
	// 	cout << image_graph[i].neighbor_capacities[0] << ' ';
	// cout << cudaGetErrorString(cudaGetLastError()) << endl;
	// initConstraints<<<numBlocks, threadsPerBlock>>>(cuda_image_graph, height, width, *K);
	// assert(cudaSuccess == cudaGetLastError());
	// printf("Initialized terminal weight values\n");

	int iteration = 1;
	while((*convergence_flag) || (!(*convergence_flag && iteration == 1))){
		for(int i = 0; i < HYPERk; i++){
			for(int j = 0; j < HYPERm; j++){
				push<<<numBlocks, threadsPerBlock>>>(cuda_image_graph, F_gpu, cuda_source, cuda_sink ,height, width, convergence_flag_gpu);
				assert(cudaSuccess == cudaGetLastError());
				printf("Local push operation %d %d\n", i, j);
				// pull<<<numBlocks, threadsPerBlock>>>(cuda_image_graph, F_gpu, cuda_source, cuda_sink, height, width);
				// assert(cudaSuccess == cudaGetLastError());
				// printf("Local pull operation\n");
				cudaMemcpy(convergence_flag, convergence_flag_gpu, sizeof(int), cudaMemcpyDeviceToHost);
				// printf("%d\n", *convergence_flag);
			}
			localRelabel<<<numBlocks, threadsPerBlock>>>(cuda_image_graph, height, width);
			assert(cudaSuccess == cudaGetLastError());
			printf("Local relabel operation\n");
		}
		// globalRelabel<<<numBlocks, threadsPerBlock>>>(cuda_image_graph, height, width, iteration);
		// assert(cudaSuccess == cudaGetLastError());
		// printf("Global relabel operation\n");
		// iteration++;
		// printf("Completed iteration %d\n\n", iteration);
		// cudaMemcpy(sink, cuda_sink, sizeof(Terminal), cudaMemcpyDeviceToHost);
		// printf("Flow: %llu\n", sink -> excess);
	}

	printf("Done with algorithm\n");
	// Load segmented image from graph using another kernel and display it

	return 0;
}

