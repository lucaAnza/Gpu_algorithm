/* Optimization-1 

-> Attualmente il programma cerca di moltiplicare *2 ogni elemento della array di array in parallelo.


*/

#include<iostream>
#include<vector>

using namespace std;




// Funzione che aggiunge il valore "index" all'indice "index" dell'array
__global__ void mul_element_for_lineIndex(int *Arr , int n){
	int id = threadIdx.x;
	
    Arr[id] = 3;
    
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
    int *GpuArr;
    int *CpuArr = new int[n];
	cudaMalloc(&GpuArr , sizeof(int) * n );  

    // Init vetture su Gpu
    size_t c = 0;
    for(int i = 0; i < nRows; i++) {
        for(int j = 0; j < vRowIndices[i].size(); j++) {
            // Copia il valore dal vettore vRowIndices al vettore GpuArr
            cudaMemcpy(&GpuArr[c], &vRowIndices[i][j], sizeof(int), cudaMemcpyHostToDevice);
            c++;
        }
    }



    mul_element_for_lineIndex<<<1,n>>>(GpuArr , n);
    // Call function
    
    
    cudaMemcpy(CpuArr , GpuArr , sizeof(int) * n , cudaMemcpyDeviceToHost );

    // Stampa di GpuArr
    cout<<"\nVettore su Cpu : "<<endl;
    for(int i=0 ; i<n ; i++){
        cout<<CpuArr[i]<<" ";
    }
    cout<<endl;
	

	cudaDeviceSynchronize();
    



	return 0;

}