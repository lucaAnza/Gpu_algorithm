/*
You are given a 1024x1024 integer matrix M. 
Each row is assigned to a thread block. 
Each thread is assigned a matrix element M[i][j]. 
It changes M[i][j] to M[i][j] + M[i][j+1] ( Exploit shared memory ) 
*/

#include<stdio.h>
#define N 100
#define M 100

//Funzione che rende ogni M[i][j] += M[i][j+1]
__global__ void strange_add(unsigned *arr){
    
    __shared__ unsigned sharedRow[M];
    int b_id = blockIdx.x;
    int t_id = threadIdx.x;
    size_t index = M * b_id + t_id;

    sharedRow[t_id] = arr[index];  //caricamento elementi riga "cpu" a "shared_memory"
    __syncthreads();
	
    if(t_id < M-1){
        sharedRow[t_id] += sharedRow[t_id + 1];
    }

    arr[index] = sharedRow[t_id];     // Copia la riga aggiornata dalla memoria condivisa alla memoria globale
    
}

// Funzione che inizializza un array a 0
__global__ void init_to_zero(unsigned *arr){

    int b_id = blockIdx.x;
    int t_id = threadIdx.x;

    size_t index = M * b_id + t_id;
	
    if(b_id < N && t_id < M){
        arr[index] = 1;
    }
    
}

int main(){
    
    unsigned *cpuMatrix,*gpuMatrix;
    cudaMalloc(&gpuMatrix , N * M * sizeof(unsigned));  
    cpuMatrix = (unsigned*) malloc( N * M * sizeof(unsigned int));

    init_to_zero<<<N,M>>>(gpuMatrix);
    cudaDeviceSynchronize();
    strange_add<<<N,M>>>(gpuMatrix);
    
    cudaMemcpy(cpuMatrix , gpuMatrix  , N*M * sizeof(unsigned int) , cudaMemcpyDeviceToHost);
    
    //stampa array cpu
    for(unsigned i = 0 ; i<N ; i++){
        for ( unsigned j = 0 ; j<M ; j++){
            printf("%2u " , cpuMatrix[ (i*M) + j]);
        }
        printf("\n");
    }
    cudaDeviceSynchronize();
    return 0;
}