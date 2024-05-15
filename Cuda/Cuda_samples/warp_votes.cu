/* Examples of use of warp votes*/


#include<stdio.h>
#include<cuda.h>


__global__ void K(){
        int id = threadIdx.x;
        printf("%d\n" , id);
        int val=2;
        //unsigned val = __any(id < 100);
        if(threadIdx.x % 32 == 0)
            printf("%d\n" , val);
}


int main(){
    printf("Start...\n");
    K<<<1,128>>>();
    cudaDeviceSynchronize();

    return 0;    
}