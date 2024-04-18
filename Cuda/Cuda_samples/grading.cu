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

#include<iostream>

using namespace std;

#define NAME_VALUATION 7

char name_valuation[] =  { 'S' , 'A' , 'B' , 'C' , 'D' , 'E' , 'U'};


// Funzione che calcola quanti voti per categoria
__global__ void find_solution(int *roll , int* voto  , int* result ,int n){
	int id = threadIdx.x;
	



	// Problema!!!    ---> Divergenza dei thread -> Esiste una soluzione miglirore ? 
	switch(voto[id]){
		case 90:
			atomicAdd(&result[0] , 1);
			break;
		case 80:
			atomicAdd(&result[1] , 1);
			break;
		case 70:
			atomicAdd(&result[2] , 1);
			break;
		case 60:
			atomicAdd(&result[3] , 1);
			break;
		case 50:
			atomicAdd(&result[4] , 1);
			break;
		case 40:
			atomicAdd(&result[5] , 1);
			break;
		case 0:
			atomicAdd(&result[6] , 1);
			break;
	}

}



// Funzione che matricola <-> voto
__global__ void init_rool(int *roll , int* voto , int n){
	int id = threadIdx.x;
	if(id<n){
		voto[id] = ((id%5)+5) * 10;
		roll[id] = id; 
	}
}
		
int main(){
	
	int n = 80; // number of students
	int *roll_gpu,*voto_gpu;
	int voto_cpu[n];
	int* result;   // Array with result [ S , A , B , C , D , E , U]
	int result_cpu[NAME_VALUATION];
	cudaMalloc(&roll_gpu , sizeof(int) * n);
	cudaMalloc(&voto_gpu , sizeof(int) * n);
	cudaMalloc(&result , sizeof(int) * size(name_valuation) );

	init_rool<<<1,n>>>(roll_gpu , voto_gpu , n);
	find_solution<<<1,n>>>(roll_gpu , voto_gpu , result , n);
	
	cudaMemcpy(voto_cpu , roll_gpu ,  sizeof(int) * n , cudaMemcpyDeviceToHost );
	cudaMemcpy(voto_cpu , voto_gpu ,  sizeof(int) * n , cudaMemcpyDeviceToHost );
	cudaMemcpy(result_cpu , result ,  sizeof(int) * NAME_VALUATION , cudaMemcpyDeviceToHost );
	cudaDeviceSynchronize();

	
	
	//stampa array finale
	for ( int i=0 ; i< size(name_valuation) ; i++){
		printf("Valuation %c = %d\n" , name_valuation[i] , result_cpu[i]);
	}

	return 0;

}