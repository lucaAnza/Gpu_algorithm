#include<stdio.h>



// Funzione che aggiunge il valore "index" all'indice "index" dell'array
__global__ void add_to_index(int *Arr , int n){
	int id = threadIdx.x;
	if(id<n){
		Arr[id] = Arr[id] + id;		
	}
}
	


// Funzione che inizializza un array a 0
__global__ void init_to_zero(int *Arr , int n ){
	int id = threadIdx.x;
	if(id<n){
		Arr[id] = id;
	}

}
		
int main(){
	

	int n = 1024;          // n = 1024  , n = 8000
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