#include <stdio.h>

__global__ void add_to_index(int *Arr, int n) {
    int id = threadIdx.x;
    if (id < n) {
        Arr[id] = Arr[id] + id;
    }
}

__global__ void init_to_zero(int *Arr, int n) {
    int id = threadIdx.x;
    if (id < n) {
        Arr[id] = 0;
    }
}

int main() {
    int n = 100;
    int *GpuArr;
    int CpuArr[n];

    cudaError_t cudaStatus;

    // Allocazione di memoria sulla GPU
    cudaStatus = cudaMalloc(&GpuArr, sizeof(int) * n);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc non riuscita: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    // Inizializza l'array sulla GPU a zero
    init_to_zero<<<1, n>>>(GpuArr, n);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Lancio del kernel fallito: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    // Aggiunge l'indice del thread ad ogni elemento
    add_to_index<<<1, n>>>(GpuArr, n);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Lancio del kernel fallito: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    // Copia il risultato indietro sulla CPU
    cudaMemcpy(CpuArr, GpuArr, sizeof(int) * n, cudaMemcpyDeviceToHost);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy fallita: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    // Stampa l'array finale
    for (int i = 0; i < n; i++) {
        printf("%d\n", CpuArr[i]);
    }

    // Libera la memoria sulla GPU
    cudaFree(GpuArr);

    return 0;
}
