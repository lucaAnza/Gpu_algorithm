/*

- Write CUDA code for the following functionality:
1. Assume following data type, filled with some values
struct Point { 
	int x,y;
}
int arr[N]; 
2. Each thread should operate on 4 elements of the array
3. Find AVG of x values.
4. If a thread sees y value above the average, it replaces all 4 y values with AVG.
Otherwise, it adds y values to a global sum.
5. Host prints the number of elements set to AVG.

*/



#include<iostream>

using namespace std;

#define N 1000000

struct Point { 
    int x,y;
};
Point arr[N]; 

__global__ void do_some_things(Point *arr , int n , int group_by , int *cont , float *sum){
    
    int id = threadIdx.x;
    int start = id*group_by;
    float sum_x = 0;
    float avg_x;
    float sum_y = 0;
    bool isAbove = false;
    
    // Calculate avg
    for(int i=start ; i<start + group_by ; i++){
        sum_x += arr[i].x;
        sum_y += arr[i].y;
    }
    avg_x = sum_x/group_by;

    // Find if some y is above the avg
    for(int i=start ; i<start + group_by && !isAbove ; i++){
        if(arr[i].y > avg_x)
            isAbove = true;
    }

    if(isAbove){
        // replaces all 4 y values with AVG
        for(int i=start ; i<start + group_by; i++){
            arr[i].y  =avg_x;
        }
        atomicAdd( cont , 1);
    }else{
        atomicAdd( sum , sum_y);
    }

}



void print_array(){
    float sum_x = 0;
    float sum_y = 0;

    for(int i=0 ; i<N ; i++){
        cout<<i<<" : ("<<arr[i].x<<" , "<<arr[i].y<<")"<<endl;
        sum_x += arr[i].x;
        sum_y += arr[i].y;
    }

    cout<<"sum ("<<sum_x<<" , "<<sum_y<<" ) \n";
}


		
int main(){
	
    int a = 4;
    int b = 1;
    int c = 12;
    int count_cpu = 0;
    float sum_cpu = 0;
    int *count_gpu;
    float *sum_gpu;
    int group_by = 32;
    for(int i=0 ; i<N ; i++){
        arr[i].x = abs(a+i+4*2 - c) ;
        arr[i].y = abs(a + b);
        c -- ; 
        a ++;
    }
    print_array();


    // Allocazione variabili GPU su memoria globale
	Point *gpuArr;
	cudaMalloc(&gpuArr , sizeof(Point) * N);
    cudaMalloc(&count_gpu , sizeof(int) * 1);
    cudaMalloc(&sum_gpu , sizeof(float) * 1);

    // Scrittura del array su GPU
    cudaMemcpy(gpuArr , arr  , N * sizeof(Point) , cudaMemcpyHostToDevice);

    // Chiamata funzione
	do_some_things<<<1,N/group_by>>>(gpuArr , N/group_by , group_by , count_gpu , sum_gpu);
    cudaDeviceSynchronize();

    // Scrittura risultati su CPU
    cudaMemcpy(arr , gpuArr , N * sizeof(Point), cudaMemcpyDeviceToHost );
    cudaMemcpy(&count_cpu , count_gpu ,sizeof(int), cudaMemcpyDeviceToHost );
    cudaMemcpy(&sum_cpu , sum_gpu , sizeof(float), cudaMemcpyDeviceToHost );
    
    //Stampa risultati finali
    cout<<"\n\nCont = "<<count_cpu<<"\nSum = "<<sum_cpu<<"\nArray after GPU computation : "<<endl;
    print_array();
    

	return 0;

}