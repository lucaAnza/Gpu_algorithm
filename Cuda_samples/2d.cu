#include<stdio.h>
#define N 5
#define M 6

// Init the matrix
__global__ void dkernel (unsigned *matrix) {

    //printf("%d , %d\n" , threadIdx.x , threadIdx.y );  //Debugging

    unsigned x = threadIdx.y;
    unsigned y = threadIdx.x;
    unsigned indice = (y * M) + x;

    matrix[indice] = indice;

}

int main(){

    dim3 block(N,M,1);
    unsigned *matrix,*hmatrix;

    cudaMalloc(&matrix , N * M * sizeof(unsigned));
    hmatrix = (unsigned*) malloc( N * M * sizeof(unsigned int));

    dkernel<<<1,block>>>(matrix);

    cudaMemcpy(hmatrix , matrix , N*M * sizeof(unsigned int) , cudaMemcpyDeviceToHost);

    for(unsigned i = 0 ; i<N ; i++){
        for ( unsigned j = 0 ; j<M ; j++){
            printf("%2d " , hmatrix[ (i*M) + j]);
        }
        printf("\n");
    }
    
    return 0;

}