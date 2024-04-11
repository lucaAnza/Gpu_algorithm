/* Optimization-1 

-> Attualmente il programma cerca di moltiplicare *2 ogni elemento della array di array in parallelo.


*/

#include<iostream>
#include<vector>

using namespace std;




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

    const int nRows = 10;   // Numero di righe dell'immagine sinistra
    const int nRightPoint = 10;  // Numero di KeyPoint dx

    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());   // Crea una matrice con nRows vettori 

    // Creazione matrice VrowIndices
    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);   

    // Init di VrowIndices
    unsigned cont = 0;
    for(int i=0 ; i<nRows ; i++){
        for(int j=0 ; j<i+1*2 ; j++){
            vRowIndices[i].push_back(j);
            cont = cont + 1;   // Da aggiungere anche al codice sorgente originale
        }
    }

    // Stampa di VrowIndices
    cout<<"Elementi delle matrice : "<<cont<<endl;
    for(int i=0 ; i<nRows ; i++){
        for(int j=0 ; j<vRowIndices[i].size() ; j++){
            cout<<vRowIndices[i][j]<<" ";
        }
        cout<<endl;
    }

	int *GpuArr;
	cudaMalloc(&GpuArr , sizeof(int) * cont );   // Allocazione vettore

    // TO-DO: Trovare un modo per capire dove inizia o finisce una riga del vettore.
	


    
    /*
    init_to_zero<<<1,n>>>(GpuArr , n);
	add_to_index<<<1,n>>>(GpuArr , n);
	cudaMemcpy(CpuArr , GpuArr , sizeof(int) * n , cudaMemcpyDeviceToHost );
	cudaDeviceSynchronize();
    */



	return 0;

}