#include<stdio.h>
#include<cuda.h>

#define BLOCKSIZE 1024
__global__ void dkernel(){
	__shared__ unsigned s;
	
	if  ( threadIdx.x == 0 ) s = 0;
	if  ( threadIdx.x == 1 ) s +=1 ;
	if  ( threadIdx.x == 31 ) s += 2;
	if  ( threadIdx.x == 0 ) printf("s=%d\n" ,s);
}
int main() {
	dkernel<<<1,BLOCKSIZE>>>();
	cudaDeviceSynchronize();
}