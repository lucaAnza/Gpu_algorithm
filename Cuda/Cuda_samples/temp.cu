#include<stdio.h>
#include<cuda.h>

#define BLOCKSIZE 10
__global__ void dkernel(){
	printf("ID : %d\n" , threadIdx.x);
}

int main() {
	dkernel<<<1,BLOCKSIZE>>>();
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(error));
	}
}
