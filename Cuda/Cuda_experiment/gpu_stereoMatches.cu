/**
* This file is part of Cuda accelerated ORB-SLAM project by Filippo Muzzini, Nicola Capodieci, Roberto Cavicchioli and Benjamin Rouxel.
 * Implemented by Filippo Muzzini.
 *
 * Based on ORB-SLAM2 (Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós) and ORB-SLAM3 (Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós)
 *
 * Project under GPLv3 Licence
*
*/

#include <cuda.h>
#include <opencv2/core/hal/interface.h>
#include <stdio.h>
#include "gpu_stereoMatches.h"

#include <vector>
#include <opencv2/core/core.hpp>

//Allocazione memoria costante in Gpu                      
__constant__  float minZ_gpu;   
__constant__  float minD_gpu;                           
__constant__  float maxD_gpu;  
__constant__  int TH_HIGH_gpu;


__global__ void cuda_test(size_t* vRowIndices_gpu , cv::KeyPoint *mvKeys_gpu , float* mDescriptors_gpu , float *mDescriptorsRight_gpu , float *mvInvScaleFactors_gpu  , float *mvScaleFactors_gpu 
                        , size_t *size_refer_gpu ) {
    
    int id = threadIdx.x;
    int b_id = blockIdx.x;

    if(id < size_refer_gpu[b_id]){
        printf("riga %d , elemento[%d] = " , b_id , id , -1 );
    }

    printf("\n\n\n");    

}


void gpu_stereoMatches(std::vector<std::vector<size_t>> vRowIndices , std::vector<cv::KeyPoint> mvKeys , float minZ , float minD , float maxD , int TH_HIGH , cv::Mat mDescriptors , cv::Mat mDescriptorsRight , 
                      std::vector<float> mvInvScaleFactors , std::vector<float> mvScaleFactors , std::vector<size_t> size_refer ){

    cv::KeyPoint *mvKeys_gpu;
    float *mvInvScaleFactors_gpu;
    float *mDescriptorsRight_gpu;
    float *mDescriptors_gpu;
    float *mvScaleFactors_gpu;
    size_t *size_refer_gpu;
    size_t *vRowIndices_gpu;
    int num_elements_left = mDescriptors.total();
    int num_elements_right = mDescriptorsRight.total();
    unsigned total_element=0;
    unsigned nRows = vRowIndices.size();

    
    // Copia parametri input in memoria costante
    cudaMemcpyToSymbol(minZ_gpu, &minZ, 1 * sizeof(float));
    cudaMemcpyToSymbol(minD_gpu, &minD, 1 * sizeof(float));
    cudaMemcpyToSymbol(maxD_gpu, &maxD, 1 * sizeof(float));
    cudaMemcpyToSymbol(TH_HIGH_gpu, &TH_HIGH, 1 * sizeof(int));

    //Allocazione memoria per array dinamici
    cudaMalloc(&mvKeys_gpu , sizeof(cv::KeyPoint) * mvKeys.size() );
    cudaMemcpy(mvKeys_gpu, mvKeys.data(), sizeof(cv::KeyPoint) * mvKeys.size(), cudaMemcpyHostToDevice); 
    cudaMalloc(&mDescriptors_gpu, num_elements_left * sizeof(float));
    cudaMemcpy(mDescriptors_gpu, (float*)mDescriptors.data, num_elements_left * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&mDescriptorsRight_gpu, num_elements_right * sizeof(float));
    cudaMemcpy(mDescriptorsRight_gpu, (float*)mDescriptorsRight.data, num_elements_right * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&mvInvScaleFactors_gpu , sizeof(float) * mvInvScaleFactors.size() );
    cudaMemcpy(mvInvScaleFactors_gpu, mvInvScaleFactors.data(), sizeof(float) * mvInvScaleFactors.size(), cudaMemcpyHostToDevice); 
    cudaMalloc(&mvScaleFactors_gpu , sizeof(float) * mvScaleFactors.size() );
    cudaMemcpy(mvScaleFactors_gpu, mvScaleFactors.data(), sizeof(float) * mvScaleFactors.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&size_refer_gpu , sizeof(size_t) * size_refer.size() );
    cudaMemcpy(size_refer_gpu, size_refer.data(), sizeof(size_t) * size_refer.size(), cudaMemcpyHostToDevice); 
    //TODO -> Evitare di fare questo ciclo e di allocare vRowIndices_temp (spreco di memoria e tempo)
    std::vector<size_t> vRowIndices_temp;
    for(int i=0 ; i<vRowIndices.size() ; i++){
        for(int j=0; j<vRowIndices[j].size() ; j++){
            total_element++;
            vRowIndices_temp.push_back(vRowIndices[i][j]);
        }
    }
    cudaMalloc(&vRowIndices_gpu , sizeof(size_t) * total_element );
    cudaMemcpy(vRowIndices_gpu, vRowIndices.data(), sizeof(size_t) * total_element, cudaMemcpyHostToDevice); 

         
    
    printf("Sto per lanciare il test della GPU by Luca Anzaldi: \n");
    //Ogni blocco rappresenta una riga di VrowIndices e ogni thread le varie colonne
    cuda_test<<<nRows,200>>>(vRowIndices_gpu , mvKeys_gpu , mDescriptors_gpu ,mDescriptorsRight_gpu , mvInvScaleFactors_gpu, mvScaleFactors_gpu , size_refer_gpu );
    cudaDeviceSynchronize();

    //Deallocazione della memoria
    cudaFree(mvKeys_gpu);
    cudaFree(mvInvScaleFactors_gpu);
    cudaFree(mDescriptorsRight_gpu);
    cudaFree(mDescriptors_gpu);
    cudaFree(mvScaleFactors_gpu);
    cudaFree(size_refer_gpu);
    cudaFree(vRowIndices_gpu);
}



/*
void gpu_stereoMatches(std::vector<cv::KeyPoint> mvKeys , float minZ , float minD , float maxD , int TH_HIGH , cv::Mat mDescriptorsRight , 
                        vector<float> mvInvScaleFactors , ORBextractor* mpORBextractorLeft , vector<float> mvScaleFactors ){
*/
