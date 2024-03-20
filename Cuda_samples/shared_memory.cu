/*
You are given a 1024x1024 integer matrix M. 
Each row is assigned to a thread block. 
Each thread is assigned a matrix element M[i][j]. 
It changes M[i][j] to M[i][j] + M[i][j+1] ( Exploit shared memory ) 
*/

#include<stdio.h>
#define N 5
#define M 300



// Funzione che inizializza un array a 0
__global__ void init_to_zero(unsigned *arr){
	
    int b_id = blockIdx.x;
    int t_id = threadIdx.x;
    size_t index = M * b_id + t_id;
	
    if(b_id < N && t_id < M){
        arr[index] = 10;
    }
    
}

int main(){
    
    unsigned *cpuMatrix,*gpuMatrix;
    cudaMalloc(&gpuMatrix , N * M * sizeof(unsigned));
    cpuMatrix = (unsigned*) malloc( N * M * sizeof(unsigned int));

 
    init_to_zero<<<N,M>>>(gpuMatrix);
    cudaDeviceSynchronize();

    cudaMemcpy(cpuMatrix , gpuMatrix  , N*M * sizeof(unsigned int) , cudaMemcpyDeviceToHost);
    

    
    for(unsigned i = 0 ; i<N ; i++){
        for ( unsigned j = 0 ; j<M ; j++){
            printf("%2u " , cpuMatrix[ (i*M) + j]);
        }
        printf("\n");
    }
    
    
    cudaDeviceSynchronize();
    

    return 0;
}