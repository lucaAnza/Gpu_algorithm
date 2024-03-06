#include<stdio.h>






// Funzione che inizializza un array a 0
__global__ void dkernel(int *Arr , int n ){
	int id = threadIdx.x;
	if(id<n){
		Arr[id] = id;
	}

}
		
int main(){
	

	int n = 2048;          // n = 1024  , n = 8000
	int *GpuArr;
	int CpuArr[n];
	cudaMalloc(&GpuArr , sizeof(int) * n);
	init_to_zero<<<1,n>>>(GpuArr , n);
	add_to_index<<<1,n>>>(GpuArr , n);
	cudaMemcpy(CpuArr , GpuArr , sizeof(int) * n , cudaMemcpyDeviceToHost );
	cudaDeviceSynchronize();

	//stampa array finale
	for ( int i=0 ; i<n ; i++){
		printf("%d\n" , CpuArr[i]);
	}

	return 0;

}