/* 

Given roll numbers and marks of 80 students in GPU programming,assign grades.

- Return an array of  [Valuation , Number of student]

- Use input arrays and output array.

|Name valuation  |Number valuation |
|----------------|-----------------|
|S|90|
|A|80|
|B|70|
|C|60|
|D|50|
|E|40|
|U|0|

- Example:

Input = [ (1,90) , (2,80) , (3,80), (4,90) , (5,0) , (6,80) , (7,80)]  
Output = [ S = 2 , A = 3 , B = 0 , ... , U = 1 ]

*/






#include<stdio.h>



// Funzione che aggiunge il valore "index" all'indice "index" dell'array
__global__ void add_to_index(int *Arr , int n){
	int id = threadIdx.x;
	if(id<n){
		Arr[id] = Arr[id] + id;		
	}
}
	


// Funzione che matricola <-> voto
__global__ void init_rool(int *arr , int n ){
	
    int id = threadIdx.x;
	if(id<n){
		//arr[id] = (id%5)+5 * 10;
        arr[id] = 5;
	}

}
		
int main(){
	

	int n = 80; // number of students
	int *GpuArr;
	int CpuArr[n];
	cudaMalloc(&GpuArr , sizeof(int) * n);

	init_rool<<<1,n>>>(GpuArr , n);
	
	cudaMemcpy(CpuArr , GpuArr , sizeof(int) * n , cudaMemcpyDeviceToHost );
	cudaDeviceSynchronize();

	//stampa array finale
	for ( int i=0 ; i<n ; i++){
		printf("%d\n" , CpuArr[i]);
	}

	return 0;

}