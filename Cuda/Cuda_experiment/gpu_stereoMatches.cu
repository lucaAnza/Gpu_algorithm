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
#include <cublas_v2.h>

#define VROWINDICES_MAX_COL 120  //TODO -> Può trasformarsi in una variabile ed essere la massima size delle righe di vrowindices.
#define MDESCRIPTOR_MAX_COL 32
#define MAX_PARTITION_FACTOR 7

//Allocazione memoria costante in Gpu                      
__constant__  float minZ_gpu;   
__constant__  float minD_gpu;                           
__constant__  float maxD_gpu;  
__constant__  int TH_HIGH_gpu;
__constant__  int mDescriptors_gpu_cols;
__constant__  int partition_factor;      //each block analize <partition_factor> line element of mDescriptor
__constant__  int mDescriptors_gpu_lines;    
__constant__  int time_calls_gpu;   // count of number of call by ComputeStereoMatches   


// Funzione che calcola la distanza tra 2 vettori
__device__ int DescriptorDistance(const unsigned char *a, const unsigned char* b){

    int dist=0;

    const int32_t* a_int = reinterpret_cast<const int32_t*>(a);
    const int32_t* b_int = reinterpret_cast<const int32_t*>(b);

    for(int i=0; i<8; i++) {
        unsigned int v = a_int[i] ^ b_int[i];
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}



__global__ void cuda_test(size_t* vRowIndices_gpu , cv::KeyPoint *mvKeys_gpu , cv::KeyPoint *mvKeysRight_gpu ,  unsigned char   * mDescriptors_gpu , unsigned char *mDescriptorsRight_gpu , float *mvInvScaleFactors_gpu  , float *mvScaleFactors_gpu 
                        , size_t *size_refer_gpu  , size_t *incremental_size_refer_gpu , int *minium_dist_gpu ) {
    
    size_t id = threadIdx.x;    // Each thread rappresent one element
    size_t b_id = blockIdx.x;   // Each block rappresent one row
    size_t num_elem_line = size_refer_gpu[b_id];
    __shared__ int minium_dist[MAX_PARTITION_FACTOR]; 

    //printf("[GPU] , size_refer_gpu[b_id=%lu]; %lu %lu \n" , b_id ,num_elem_line , size_refer_gpu[b_id]);    //DEBUG 

    if(  (id < num_elem_line) ){

        
        //printf("ID(%lu) < num_elem(%lu) of line %lu  mDescript_lines(%d) part_Fact(%d)\n" , id , num_elem_line , b_id , mDescriptors_gpu_lines , partition_factor);

        int begin = (int)b_id * partition_factor;

        for(int iL= begin , partition_i = 0; (iL<begin + partition_factor) && (iL<mDescriptors_gpu_lines) ; iL++ , partition_i++){
            
            minium_dist[partition_i] = TH_HIGH_gpu;    // Serve ancora?

            //Pre-For-Loop////////////////////////////////////////////////////////////////////////////////////////////////////
            const cv::KeyPoint &kpL = mvKeys_gpu[iL];
            const int &levelL = kpL.octave;
            const float &vL = kpL.pt.y;
            const float &uL = kpL.pt.x;

            
            //const vector<size_t> &vCandidates = vRowIndices_gpu[vL];      //TODELETE  

            //if(vCandidates.empty())                                       //TODELETE  
            //    return;  // Terminate thread                              //TODELETE

            
            const float minU = uL-maxD_gpu;
            const float maxU = uL-minD_gpu;

            if(maxU<0)
                return;

            int bestDist = TH_HIGH_gpu;
            size_t bestIdxR = 0;

            //const cv::Mat &dL = mDescriptors_gpu.row(iL);   // TODELETE

            //For-Loop////////////////////////////////////////////////////////////////////////////////////////////////////
            const size_t index_taked = incremental_size_refer_gpu[(int)vL] - size_refer_gpu[(int)vL];
            const size_t iR = vRowIndices_gpu[incremental_size_refer_gpu[(int)vL] - size_refer_gpu[(int)vL] + (id + partition_i)] ;
            const cv::KeyPoint &kpR = mvKeysRight_gpu[iR];

            
            // Working debug   (check if iL iR GPU is the same of CPU use-> grep "\{1\}.*iL\[820\] iR\[694\]" output.txt)
            //printf("{%d}[GPU]elements of mvKeys[iL].pt.y(vL) : %f ,  iL[%d] iR[%lu]  take index : %lu i_sr = %lu  sr = %lu \n" , time_calls_gpu , mvKeys_gpu[iL].pt.y  ,iL , iR , index_taked , incremental_size_refer_gpu[(int)vL] , size_refer_gpu[(int)vL]); 


            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)  
                return;
            
            const float &uR = kpR.pt.x;  

            if(uR>=minU && uR<=maxU) {   // Controllo se la x del keypointCandidatoDX sta in un range
            
                //const cv::Mat &dR = mDescriptorsRight.row(iR);     //TODELETE

                const unsigned char *dL =  (mDescriptors_gpu + mDescriptors_gpu_cols * iL );
                const unsigned char *dR =  (mDescriptorsRight_gpu + mDescriptors_gpu_cols * iR );
                const int dist = DescriptorDistance(dL , dR); 
                
                //atomicMin( &minium_dist[partition_i] , dist);          // TODO : Check if it is correct
                __syncthreads(); // Waiting that all thread of the block calculate the distance
                 
                //minium_dist_gpu[iL] = dist;
                //printf("{%d} [GPU] Distanza minimima della linea iL(%d) = %d  clt-bID(%lu)tID(%lu)\n" , time_calls_gpu , iL , minium_dist[partition_i] , b_id , id);

                
                
                

                /*     

                A   A   A   A 
                |   |   |   |
                |   |   |   |
                |   |   |   | 
                
                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
                
                //CONTINUE FROM HERE...

                */

               printf("{%d}[GPU]dist of element iL[%d] iR[%lu] : %d \n" , time_calls_gpu , iL , iR , dist); 
                

                
                
                
            }
        }
        

    }


}


void gpu_stereoMatches(int time_calls , std::vector<std::vector<size_t>> vRowIndices , std::vector<cv::KeyPoint> mvKeys , std::vector<cv::KeyPoint> mvKeysRight , float minZ , float minD , float maxD , int TH_HIGH , cv::Mat mDescriptors , cv::Mat mDescriptorsRight , 
                      std::vector<float> mvInvScaleFactors , std::vector<float> mvScaleFactors , std::vector<size_t> size_refer){

    cv::KeyPoint *mvKeys_gpu;
    cv::KeyPoint *mvKeysRight_gpu;
    float *mvInvScaleFactors_gpu;
    unsigned char *mDescriptorsRight_gpu;
    unsigned char *mDescriptors_gpu;
    float *mvScaleFactors_gpu;
    size_t *size_refer_gpu;                  // Vettore che associa ogni elemento N al numero di colonne del vettore vRowIndices[N]
    size_t *incremental_size_refer_gpu;      // Vettore che associa ogni elemento N alla somma del numero di colonne del vettore fino a vRowIndices[N]
    size_t *vRowIndices_gpu;
    int num_elements_left = mDescriptors.total();
    int num_elements_right = mDescriptorsRight.total();
    unsigned total_element=0;
    unsigned nRows = vRowIndices.size();
    unsigned N = mvKeys.size();
    int *minium_dist_gpu;
    
    // Copia parametri input in memoria costante
    cudaMemcpyToSymbol(minZ_gpu, &minZ, 1 * sizeof(float));
    cudaMemcpyToSymbol(minD_gpu, &minD, 1 * sizeof(float));
    cudaMemcpyToSymbol(maxD_gpu, &maxD, 1 * sizeof(float));
    cudaMemcpyToSymbol(TH_HIGH_gpu, &TH_HIGH, 1 * sizeof(int));
    cudaMemcpyToSymbol(mDescriptors_gpu_cols, &mDescriptors.cols, 1 * sizeof(int));
    int part_const = ((int)N/nRows) + 1;
    cudaMemcpyToSymbol(partition_factor, &part_const, 1 * sizeof(int));
    cudaMemcpyToSymbol(mDescriptors_gpu_lines, &num_elements_left, 1 * sizeof(int));
    cudaMemcpyToSymbol(time_calls_gpu, &time_calls, 1 * sizeof(int));
    

    //Allocazione memoria per array dinamici
    cudaMalloc(&mvKeys_gpu , sizeof(cv::KeyPoint) * mvKeys.size() );
    cudaMemcpy(mvKeys_gpu, mvKeys.data(), sizeof(cv::KeyPoint) * mvKeys.size(), cudaMemcpyHostToDevice); 
    cudaMalloc(&mvKeysRight_gpu , sizeof(cv::KeyPoint) * mvKeysRight.size() );
    cudaMemcpy(mvKeysRight_gpu, mvKeysRight.data(), sizeof(cv::KeyPoint) * mvKeysRight.size(), cudaMemcpyHostToDevice); 
    cudaMalloc(&mDescriptors_gpu, num_elements_left * sizeof(unsigned char));
    cudaMemcpy(mDescriptors_gpu, (unsigned char*)mDescriptors.data, num_elements_left * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMalloc(&mDescriptorsRight_gpu, num_elements_right * sizeof(unsigned char));
    cudaMemcpy(mDescriptorsRight_gpu, (unsigned char*)mDescriptorsRight.data, num_elements_right * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMalloc(&mvInvScaleFactors_gpu , sizeof(float) * mvInvScaleFactors.size() );
    cudaMemcpy(mvInvScaleFactors_gpu, mvInvScaleFactors.data(), sizeof(float) * mvInvScaleFactors.size(), cudaMemcpyHostToDevice); 
    cudaMalloc(&mvScaleFactors_gpu , sizeof(float) * mvScaleFactors.size() );
    cudaMemcpy(mvScaleFactors_gpu, mvScaleFactors.data(), sizeof(float) * mvScaleFactors.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&size_refer_gpu , sizeof(size_t) * size_refer.size() );
    cudaMemcpy(size_refer_gpu, size_refer.data(), sizeof(size_t) * size_refer.size(), cudaMemcpyHostToDevice); 


    //TODO -> Evitare di fare questo ciclo e di allocare vRowIndices_temp (spreco di memoria e tempo) OR eseguirlo in GPU
    std::vector<size_t> vRowIndices_temp;
    std::vector<size_t> incremental_size_refer;
    incremental_size_refer.resize(size_refer.size());
    printf("vrowindices.size() %d\n" , vRowIndices.size() );
    for(int i=0 ; i<vRowIndices.size() ; i++){

        if(i>0)
            incremental_size_refer[i] = incremental_size_refer[i-1] + size_refer[i];
        else
            incremental_size_refer[i] = size_refer[i];

        for(int j=0; j<vRowIndices[i].size() ; j++){
            total_element++;
            vRowIndices_temp.push_back(vRowIndices[i][j]);
        }
    }

    //printf of size_refer and incremental_size_refer
    
    for(int i=0 ; i<vRowIndices.size() ; i++){
        printf("%lu \t %lu\n" , size_refer[i] , incremental_size_refer[i]);
    }

    cudaMalloc(&incremental_size_refer_gpu , sizeof(size_t) * incremental_size_refer.size() );
    cudaMemcpy(incremental_size_refer_gpu, incremental_size_refer.data(), sizeof(size_t) * size_refer.size(), cudaMemcpyHostToDevice); 
    cudaMalloc(&vRowIndices_gpu , sizeof(size_t) * total_element );
    cudaMemcpy(vRowIndices_gpu, vRowIndices_temp.data(), sizeof(size_t) * total_element, cudaMemcpyHostToDevice); 
    

    //Test - TODELETE
    cudaMalloc(&minium_dist_gpu , sizeof(int) * N);


    printf("Sto per lanciare il test della GPU by Luca Anzald: \n");
    cuda_test<<<nRows,VROWINDICES_MAX_COL>>>(vRowIndices_gpu , mvKeys_gpu , mvKeysRight_gpu , mDescriptors_gpu ,mDescriptorsRight_gpu , mvInvScaleFactors_gpu, mvScaleFactors_gpu , size_refer_gpu , incremental_size_refer_gpu , minium_dist_gpu );
    cudaDeviceSynchronize();

    //Test -TODELETE
    int distanzaMinime[N];
    cudaMemcpy(distanzaMinime, minium_dist_gpu, sizeof(int) * N, cudaMemcpyDeviceToHost);
    printf("{%d}Stampa delle distanze minime : \n" , time_calls);
    for(int i=0 ; i<N ; i++){
        printf("{%d} %d : %d\n" , time_calls ,  i , distanzaMinime[i]);
    }


    //Deallocazione della memoria
    cudaFree(mvKeys_gpu);
    cudaFree(mvKeysRight_gpu);
    cudaFree(mvInvScaleFactors_gpu);
    cudaFree(mDescriptorsRight_gpu);
    cudaFree(mDescriptors_gpu);
    cudaFree(mvScaleFactors_gpu);
    cudaFree(size_refer_gpu);
    cudaFree(incremental_size_refer_gpu);
    cudaFree(vRowIndices_gpu);

}



/*
void gpu_stereoMatches(std::vector<cv::KeyPoint> mvKeys , float minZ , float minD , float maxD , int TH_HIGH , cv::Mat mDescriptorsRight , 
                        vector<float> mvInvScaleFactors , ORBextractor* mpORBextractorLeft , vector<float> mvScaleFactors ){
*/


