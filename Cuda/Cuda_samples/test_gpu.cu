#include <stdio.h>
#include <cuda.h>

void check_cuda_error(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void add_vectors(int *a, int *b, int *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(void) {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int n = 10;
    int size = n * sizeof(int);

    // Allocate memory on host
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Check if memory allocation on host was successful
    if (a == NULL || b == NULL || c == NULL) {
        printf("Failed to allocate memory on host.\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory on device
    check_cuda_error(cudaMalloc((void **)&d_a, size));
    check_cuda_error(cudaMalloc((void **)&d_b, size));
    check_cuda_error(cudaMalloc((void **)&d_c, size));

    // Initialize vectors on host
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Copy vectors from host to device
    check_cuda_error(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    // Launch kernel on device
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    add_vectors<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, n);

    // Check for errors in kernel launch
    check_cuda_error(cudaGetLastError());
    check_cuda_error(cudaDeviceSynchronize());

    // Copy result from device to host
    check_cuda_error(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    // Print result on host
    for (int i = 0; i < n; i++) {
        printf("c[%d] = %d\n", i, c[i]);
    }

    // Free memory on device
    check_cuda_error(cudaFree(d_a));
    check_cuda_error(cudaFree(d_b));
    check_cuda_error(cudaFree(d_c));

    // Free memory on host
    free(a);
    free(b);
    free(c);

    return 0;
}

