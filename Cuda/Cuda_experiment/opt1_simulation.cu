/* Optimization-1 
-> Attualmente il programma cerca di moltiplicare ogni elemento di una riga per l'indice della riga+1 in parallelo.

-> Per compilare: "nvcc -lcublas opt1_simulation.cu"
*/

#include<iostream>
#include<vector>
#include <cublas_v2.h>

using namespace std;


// Funzione che moltiplica ogni elemento di una riga per l'indice di riga + 1
__global__ void mul_element_for_lineIndex(float *arr , int const *index_refer,  int n){
	int id = threadIdx.x;
	
    arr[id] = arr[id] * index_refer[id];
}
	
	
int main(){

    const int nRows = 4;   // Numero di righe dell'immagine sinistra
    //vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());   // Crea una matrice con nRows vettori 

    // Creazione matrice VrowIndices
    vector<vector<size_t> > vRowIndices = {
        {3, 7, 15},
        {5, 6, 12, 22},
        {2, 8, 9},
        {6, 7}
    };
    //Ampliamento matrice VrowIndices
    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);   

    // Stampa di VrowIndices 
    cout<<"\nVRowIndices : "<<endl;
    int n=0;
    for(int i=0 ; i<nRows ; i++){
        for(int j=0 ; j<vRowIndices[i].size() ; j++){
            cout<<vRowIndices[i][j]<<" ";
            n++;
        }
        cout<<endl;
    }
    
    // Allocazione vettore su Gpu e Cpu
    float *GpuArr;
    float *CpuArr = new float[n];
    int *index_refer_gpu;
    int *index_refer_cpu = new int[n];
	cudaMalloc(&GpuArr , sizeof(float) * n );  
    cudaMalloc(&index_refer_gpu , sizeof(int) * n );  


    // Init vettore su Gpu
    size_t c = 0;
    for(int i = 0; i < nRows; i++) {
        for(int j = 0; j < vRowIndices[i].size(); j++) {
            // Copia il valore dal vettore vRowIndices al vettore GpuArr
            float temp = (float) vRowIndices[i][j];
            cudaMemcpy(&GpuArr[c], &temp , sizeof(float), cudaMemcpyHostToDevice);
            index_refer_cpu[c] = i+1;
            c++;
        }
    }
    cudaMemcpy(index_refer_gpu, index_refer_cpu, sizeof(int) * n, cudaMemcpyHostToDevice);


    // Chiamata funzione + spostamento gpu -> cpu
    mul_element_for_lineIndex<<<1,n>>>(GpuArr , index_refer_gpu , n);
    cudaMemcpy(CpuArr , GpuArr , sizeof(float) * n , cudaMemcpyDeviceToHost );
    cudaMemcpy(index_refer_cpu , index_refer_gpu , sizeof(int) * n , cudaMemcpyDeviceToHost );

    // Stampa di GpuArr
    cout<<"\nVettore su Cpu : "<<endl;
    for(int i=0 ; i<n ; i++){
        cout<<CpuArr[i]<<" ";
    }
    // Stampa di Index refer
    cout<<"\nIndex refer: "<<endl;
    for(int i=0 ; i<n ; i++){
        cout<<index_refer_cpu[i]<<" ";
    }
    cout<<endl;


    // Calcolo del minimo per ogni array
    int offset = 0;
    int result;
    int size;
    int *minium_array = new int[nRows];
    cublasHandle_t handle;
    for(int i=0 ; i<nRows ; i++){
        cublasCreate(&handle);
        size = vRowIndices[i].size();
        cublasIsamin(handle, size , GpuArr + offset, 1, &result);
        offset += size;
        std::cout << "\nMinimum element of line "<<i<<"is at index: " << result << std::endl;
        minium_array[i] = result;
        cublasDestroy(handle);
    }

    // Sort of minium_array

	cudaDeviceSynchronize();
    
	return 0;

}