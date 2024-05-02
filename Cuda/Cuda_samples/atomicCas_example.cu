#include<stdio.h>
#include<cuda.h>

__global__ void k1(int *gg){

    //Print-1 
	int old = atomicCAS(gg,0,threadIdx.x + 1);
	if (old == 0){
		printf("Thread %d succeeded 1.\n" , threadIdx.x);
	}
    //Print-2 
	old = atomicCAS(gg,0,threadIdx.x + 1);
	if (old == 0){
		printf("Thread %d succeeded 2. \n" , threadIdx.x);
	}
    //Print-3 
	old = atomicCAS(gg,0,threadIdx.x - 1);
	if (old == 0){
		printf("Thread %d succeeded 3.\n" , threadIdx.x);
	}

}

int main(){
	int *gg;
	cudaMalloc(&gg , sizeof(int));
	cudaMemset(&gg , 0 , sizeof(int) );
	k1<<<2,32>>>(gg);
	cudaDeviceSynchronize();
	return 0;
}