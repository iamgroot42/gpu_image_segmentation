#include <iostream>

#define N 256
#define M 256

using namespace std;

__global__ void pushKernel(long long *globalExcess, long long *globalLabels)
{
	__shared__ long long labels[M * N];
	__shared__ long long excess[M * N];
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	excess[i] = globalExcess[i];
	labels[i] = globalLabels[i];
	__syncthreads();

	
}
__global__ void pullKernel(int u);
__global__ void localRelabel(int u);
__global__ void globalRelabel(int u); 

int main()
{

	return 0;
}