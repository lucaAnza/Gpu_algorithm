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

#define VROWINDICES_MAX_COL 120  //TODO -> Può trasformarsi in una variabile ed essere la massima size delle righe di vrowindices.
#define MDESCRIPTOR_MAX_COL 32

//Allocazione memoria costante in Gpu                      
__constant__  float minZ_gpu;   
__constant__  float minD_gpu;                           
__constant__  float maxD_gpu;  
__constant__  int TH_HIGH_gpu;
__constant__  int mDescriptors_gpu_cols;
__constant__  int partition_factor;      //each block analize <partition_factor> line element of mDescriptor


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
                        , size_t *size_refer_gpu  , size_t *incremental_size_refer_gpu) {
    
    size_t id = threadIdx.x;    // Each thread rappresent one element
    size_t b_id = blockIdx.x;   // Each block rappresent one row
    size_t num_elem = size_refer_gpu[b_id];
    size_t index;

    if(  (id < size_refer_gpu[b_id]) ){

        /*
        for(int i=0 ; i<partition_factor ; i++){
            // TODO -> fare in modo di non lasciare fuori gli ultimi elementi
        }
        */

        index = ((incremental_size_refer_gpu[b_id] - num_elem) + id);

        //Pre-For-Loop////////////////////////////////////////////////////////////////////////////////////////////////////
        const cv::KeyPoint &kpL = mvKeys_gpu[b_id];
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
        const size_t iR = vRowIndices_gpu[incremental_size_refer_gpu[(int)vL] - size_refer_gpu[(int)vL] + id] ;
        const cv::KeyPoint &kpR = mvKeysRight_gpu[iR];   

        if(kpR.octave<levelL-1 || kpR.octave>levelL+1)  
            return;
        
        const float &uR = kpR.pt.x;  

        if(uR>=minU && uR<=maxU) {   // Controllo se la x del keypointCandidatoDX sta in un range
        
            //const cv::Mat &dR = mDescriptorsRight.row(iR);     //TODELETE

            const unsigned char *dL =  (mDescriptors_gpu + mDescriptors_gpu_cols * b_id );
            const unsigned char *dR =  (mDescriptorsRight_gpu + mDescriptors_gpu_cols * iR );
            const int dist = DescriptorDistance(dL , dR);   
            
            ////////////////////////////
            ////////// TO DO ///////////
            ////////////////////////////
            // Testare se il valore dist è corretto ed è uguale a quello generato sulla CPU
            // Capire discrepanza tra il valore di N[GPU] e N[CPU] perchè è diverso??? 32 vs 2000
            //////////////////////////// 
            ////////// TO DO ///////////
            ////////////////////////////

            printf("[GPU]dist of element iL[%lu] iR[%lu] : %d \n" , b_id , iR , dist); 

            /*                               CONTINUE FROM HERE...
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdxR = iR;
            }
            */
        }
        
        

    }


}


void gpu_stereoMatches(std::vector<std::vector<size_t>> vRowIndices , std::vector<cv::KeyPoint> mvKeys , std::vector<cv::KeyPoint> mvKeysRight , float minZ , float minD , float maxD , int TH_HIGH , cv::Mat mDescriptors , cv::Mat mDescriptorsRight , 
                      std::vector<float> mvInvScaleFactors , std::vector<float> mvScaleFactors , std::vector<size_t> size_refer ){

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

    
    // Copia parametri input in memoria costante
    cudaMemcpyToSymbol(minZ_gpu, &minZ, 1 * sizeof(float));
    cudaMemcpyToSymbol(minD_gpu, &minD, 1 * sizeof(float));
    cudaMemcpyToSymbol(maxD_gpu, &maxD, 1 * sizeof(float));
    cudaMemcpyToSymbol(TH_HIGH_gpu, &TH_HIGH, 1 * sizeof(int));
    cudaMemcpyToSymbol(mDescriptors_gpu_cols, &mDescriptors.cols, 1 * sizeof(int));
    int part_const = (int)N/nRows;
    cudaMemcpyToSymbol(partition_factor, &part_const, 1 * sizeof(int));

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
        incremental_size_refer[i] = 0;
        if(i>0)
            incremental_size_refer[i] = incremental_size_refer[i] + size_refer[i-1];

        for(int j=0; j<vRowIndices[i].size() ; j++){
            total_element++;
            vRowIndices_temp.push_back(vRowIndices[i][j]);
        }
    }
    cudaMalloc(&incremental_size_refer_gpu , sizeof(size_t) * incremental_size_refer.size() );
    cudaMemcpy(incremental_size_refer_gpu, incremental_size_refer.data(), sizeof(size_t) * size_refer.size(), cudaMemcpyHostToDevice); 
    cudaMalloc(&vRowIndices_gpu , sizeof(size_t) * total_element );
    cudaMemcpy(vRowIndices_gpu, vRowIndices_temp.data(), sizeof(size_t) * total_element, cudaMemcpyHostToDevice); 

    printf("numero colonne di mDescriptor : %d\n" , mDescriptors.cols);

    printf("Sto per lanciare il test della GPU by Luca Anzald: \n");
    //Ogni blocco rappresenta una riga di VrowIndices e ogni thread le varie colonne
    cuda_test<<<nRows,VROWINDICES_MAX_COL>>>(vRowIndices_gpu , mvKeys_gpu , mvKeysRight_gpu , mDescriptors_gpu ,mDescriptorsRight_gpu , mvInvScaleFactors_gpu, mvScaleFactors_gpu , incremental_size_refer_gpu , size_refer_gpu );
    cudaDeviceSynchronize();

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


