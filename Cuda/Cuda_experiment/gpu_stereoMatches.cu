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


__global__ void cuda_test(cv::KeyPoint *mvKeys_gpu) {
    
    int temp = threadIdx.x;

    printf("%f , %f , %f \n" , minZ_gpu , minD_gpu , maxD_gpu );
    printf("punto prova di mvKeys[0] : %f \n\n\n" , mvKeys_gpu->pt.x);

}


void gpu_stereoMatches(std::vector<cv::KeyPoint> mvKeys , float minZ , float minD , float maxD  ){

    cv::KeyPoint *mvKeys_gpu;
    
    // Copia parametri input in memoria costante
    cudaMemcpyToSymbol(minZ_gpu, &minZ, 1 * sizeof(float));
    cudaMemcpyToSymbol(minD_gpu, &minD, 1 * sizeof(float));
    cudaMemcpyToSymbol(maxD_gpu, &maxD, 1 * sizeof(float));

    //Allocazione memoria per array dinamici
    cudaMalloc(&mvKeys_gpu , sizeof(cv::KeyPoint) * mvKeys.size() );
    cudaMemcpy(mvKeys_gpu, mvKeys.data(), sizeof(cv::KeyPoint) * mvKeys.size(), cudaMemcpyHostToDevice); 


    printf("Sto per lanciare il test della GPU by Luca Anzaldi: \n");
    cuda_test<<<1,1>>>(mvKeys_gpu);
}



/*

void gpu_stereoMatches(std::vector<cv::KeyPoint> mvKeys , float minZ , float minD , float maxD , int TH_HIGH , cv::Mat mDescriptorsRight , 
                        vector<float> mvInvScaleFactors , ORBextractor* mpORBextractorLeft , vector<float> mvScaleFactors ){

*/
