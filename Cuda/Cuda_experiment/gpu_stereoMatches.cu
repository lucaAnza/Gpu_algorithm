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
#define NUM_THREAD 256

//Allocazione memoria costante in Gpu                      
__constant__  float minZ_gpu;   
__constant__  float minD_gpu;                           
__constant__  float maxD_gpu;  
__constant__  int TH_HIGH_gpu;
__constant__  int mDescriptors_gpu_cols;
__constant__  int partition_factor;      //each block analize <partition_factor> line element of mDescriptor
__constant__  int mDescriptors_gpu_lines;    
__constant__  int time_calls_gpu;   // count of number of call by ComputeStereoMatches   
__constant__  int thOrbDist_gpu;   // count of number of call by ComputeStereoMatches 
__constant__  int L_gpu; 
__constant__  int w_gpu;
__constant__  int Nd;
__constant__  float mbf_gpu;



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



// Funzione che calcola le distanze minime tra due i descrittori di 2 punti chiave.
__global__ void findMiniumDistance(size_t* vRowIndices_gpu , cv::KeyPoint *mvKeys_gpu , cv::KeyPoint *mvKeysRight_gpu ,  unsigned char   * mDescriptors_gpu , unsigned char *mDescriptorsRight_gpu , float *mvInvScaleFactors_gpu  , float *mvScaleFactors_gpu 
                        , size_t *size_refer_gpu  , size_t *incremental_size_refer_gpu , int *miniumDist_gpu , size_t *miniumDistIndex_gpu ) {
    
    size_t id = threadIdx.x;    // Each thread rappresent one element
    size_t b_id = blockIdx.x;   // Each block rappresent n (n = partion_factor) row
    __shared__ int miniumDistShared[MAX_PARTITION_FACTOR]; 
    int begin = (int)b_id * partition_factor;

    // Init of minium distance vector
    for(int iL= begin , partition_i = 0; (iL<begin + partition_factor) && (iL<mDescriptors_gpu_lines) ; iL++ , partition_i++){
        miniumDistShared[partition_i] = TH_HIGH_gpu;
        miniumDist_gpu[iL] = TH_HIGH_gpu;
        miniumDistIndex_gpu[iL] = 0;
    }
    
    __syncthreads();

    for(int iL= begin , partition_i = 0; (iL<begin + partition_factor) && (iL<mDescriptors_gpu_lines) ; iL++ , partition_i++){
        const cv::KeyPoint &kpL = mvKeys_gpu[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;        
        const float minU = uL-maxD_gpu;
        const float maxU = uL-minD_gpu;

        if(maxU<0){
            continue;
        }

        //For-Loop////////////////////////////////////////////////////////////////////////////////////////////////////
        size_t num_elem_line = size_refer_gpu[(int)vL];
        const size_t index_taked = incremental_size_refer_gpu[(int)vL] - num_elem_line;
        
        if(id < num_elem_line){
            const size_t iR = vRowIndices_gpu[incremental_size_refer_gpu[(int)vL] - num_elem_line + (id)] ;
            const cv::KeyPoint &kpR = mvKeysRight_gpu[iR];
            
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;
            
            const float &uR = kpR.pt.x;  
            if(uR>=minU && uR<=maxU) {   // Controllo se la x del keypointCandidatoDX sta in un range
                const unsigned char *dL =  (mDescriptors_gpu + mDescriptors_gpu_cols * iL );
                const unsigned char *dR =  (mDescriptorsRight_gpu + mDescriptors_gpu_cols * iR );
                const int dist = DescriptorDistance(dL , dR); 
                atomicMin( &miniumDistShared[partition_i] , dist);     
            }
        }
    }
    __syncthreads();

    // Save minium distance on Arrays
    for(int iL= begin , partition_i = 0; (iL<begin + partition_factor) && (iL<mDescriptors_gpu_lines) ; iL++ , partition_i++){
        //Save of the value
        miniumDist_gpu[iL] = miniumDistShared[partition_i];
        
        //Save of the index ->  TODO(not urgent) -> Try to optimize this 
        const cv::KeyPoint &kpL = mvKeys_gpu[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;        
        const float minU = uL-maxD_gpu;
        const float maxU = uL-minD_gpu;
        if(maxU<0){
            continue;
        }
        size_t num_elem_line = size_refer_gpu[(int)vL];
        const size_t index_taked = incremental_size_refer_gpu[(int)vL] - num_elem_line;
        if(id < num_elem_line){
            const size_t iR = vRowIndices_gpu[incremental_size_refer_gpu[(int)vL] - num_elem_line + (id)] ;
            const cv::KeyPoint &kpR = mvKeysRight_gpu[iR];
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;
            const float &uR = kpR.pt.x;  
            if(uR>=minU && uR<=maxU) {   // Controllo se la x del keypointCandidatoDX sta in un range
                const unsigned char *dL =  (mDescriptors_gpu + mDescriptors_gpu_cols * iL );
                const unsigned char *dR =  (mDescriptorsRight_gpu + mDescriptors_gpu_cols * iR );
                const int dist = DescriptorDistance(dL , dR); 
                if(dist == miniumDistShared[partition_i])
                    miniumDistIndex_gpu[iL] = iR;
            }
        }
    }
    

}


// Calculate Norm1 of 2 vector with the same lenght
__device__ float norm1(const uchar *V1 , const uchar* V2 , int size , int i1 , int j1 , int i2 , int j2 , int cols1 , int cols2 , int incR , int iL){

    float sum = 0;
    int countRow = 0;
    int countCol = 0;
    int j1_temp = j1;
    int j2_temp = j2;
    while(countRow < size){
        while(countCol < size){
            int index1 = ( (i1*cols1) + j1 );
            int index2 = ( (i2*cols2) + j2 );
            sum = sum + abs(((float) V1[index1] - (float) V2[index2]));
            countCol++;
            j1++;
            j2++;
        }
        i1++;
        i2++;
        j1 = j1_temp;
        j2 = j2_temp;
        countCol = 0;
        countRow++;
    }

    return sum;
    
}


__global__ void slidingWindow( int rows , int cols , float *scaleFactors , uchar *d_images  , uchar *d_inputImage , int rows_r , int cols_r , 
                               float *scaleFactorsRight, uchar *d_imagesR , uchar *d_inputImageR ,cv::KeyPoint *mvKeys_gpu , cv::KeyPoint *mvKeysRight_gpu , float *mvInvScaleFactors_gpu  , float *mvScaleFactors_gpu , int *miniumDist_gpu , size_t *miniumDistIndex_gpu , float *IL , float *IR , 
                               int* vDistIdx_gpu , float *mvDepth_gpu , float *mvuRight_gpu  ){

    extern __shared__ int vDists[];  
    size_t id = threadIdx.x;   
    size_t b_id = blockIdx.x;   
    size_t iL = (b_id * blockDim.x) + id;  // Index is iL of CPU function (iL [0 -> 2026])
    
    int bestDist = INT_MAX;
    int bestincR = 0;    
    int distN[11]; // TODO  -> Is static could be a problem
    bool exclude = false;
    vDistIdx_gpu[iL] = -1;
    mvuRight_gpu[iL] = -1;
    mvDepth_gpu[iL] = -1;

    if(iL < Nd && miniumDist_gpu[iL] < thOrbDist_gpu ){  //if(bestDist<thOrbDist)

        // coordinates in image pyramid at keypoint scale
        const cv::KeyPoint &kpL = mvKeys_gpu[iL];
        size_t bestIdxR = miniumDistIndex_gpu[iL];
        const float uR0 = mvKeysRight_gpu[bestIdxR].pt.x;        
        const float scaleFactor = mvInvScaleFactors_gpu[kpL.octave];   
        const float scaleduL = round(kpL.pt.x*scaleFactor);     
        const float scaledvL = round(kpL.pt.y*scaleFactor);     
        const float scaleduR0 = round(uR0*scaleFactor);         
        int i_startIL = scaledvL-w_gpu;
        int i_finalIL = scaledvL+w_gpu+1;
        int j_startIL = scaleduL-w_gpu;
        int j_finalIL = scaleduL+w_gpu+1;
        const float d_scaleFactor = scaleFactors[kpL.octave];  
        const float d_scaleFactorRight = scaleFactorsRight[kpL.octave]; 
        const uint new_rows = round(rows * 1/d_scaleFactor);
        const uint new_cols = round(cols * 1/d_scaleFactor);
        const uint new_rows_r  = round(rows_r * 1/d_scaleFactorRight);
        const uint new_cols_r = round(cols_r * 1/d_scaleFactorRight);
        const float iniu = scaleduR0+L_gpu-w_gpu;       
        const float endu = scaleduR0+L_gpu+w_gpu+1;

        
        if(iniu<0 || endu >= new_cols_r )   // per evitare di uscire dai range
            exclude = true;
        
        
        for(int incR=-L_gpu , count = 0 ; incR<=+L_gpu && !exclude ; incR++ , count++ ){

            int i_startIR = scaledvL-w_gpu;
            int i_finalIR = scaledvL+w_gpu+1;
            int j_startIR = scaleduR0+incR-w_gpu;
            int j_finalIR = scaleduR0+incR+w_gpu+1;
            int level = kpL.octave;
            int offset_level = (rows * cols) * level;
            int offset_levelR = (rows_r * cols_r) * level;
            int line_size = ( (w_gpu*2) + 1 );
            int col_offset = new_cols;
            int col_offsetRight = new_cols_r;
            int block_dim = 16;
            uchar *imgPyramid , *imgPyramidRight;
            if(level == 0){
                imgPyramid = d_inputImage;
                imgPyramidRight = d_inputImageR;
            }else{
                imgPyramid = d_images;
                imgPyramidRight = d_imagesR;
            }

            
            int dist = (int) norm1( imgPyramid+offset_level , imgPyramidRight+offset_levelR , line_size , i_startIL , j_startIL , i_startIR , j_startIR , new_cols , new_cols_r , incR , iL);
            distN[count] = dist;
            
            if(dist<bestDist)
            {
                bestDist =  dist;
                bestincR = incR;
            }
            
        }

    }

    __syncthreads();

    if(iL < Nd && miniumDist_gpu[iL] < thOrbDist_gpu ){

        // TODO -> this variable are already on the previous if.  you can optimaze
        // coordinates in image pyramid at keypoint scale
        const cv::KeyPoint &kpL = mvKeys_gpu[iL];
        const float &uL = kpL.pt.x;
        size_t bestIdxR = miniumDistIndex_gpu[iL];
        const float uR0 = mvKeysRight_gpu[bestIdxR].pt.x;        
        const float scaleFactor = mvInvScaleFactors_gpu[kpL.octave];      
        const float scaleduR0 = round(uR0*scaleFactor); 
 
        if(bestincR==-L_gpu || bestincR==L_gpu)
            return;
        
        
        const float dist1 = distN[L_gpu+bestincR-1];
        const float dist2 = distN[L_gpu+bestincR];
        const float dist3 = distN[L_gpu+bestincR+1];  

    
        const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

        if(deltaR<-1 || deltaR>1)
            return;
            

        float bestuR = mvScaleFactors_gpu[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

        float disparity = (uL-bestuR);  // TODO : disparity has an absolute error < 0.01 , try to fix!

        if(disparity>=minD_gpu && disparity<maxD_gpu)
        {
            if(disparity<=0)
            {
                disparity=0.01;
                bestuR = uL-0.01;
            }
            mvDepth_gpu[iL] = mbf_gpu/disparity;
            mvuRight_gpu[iL] = bestuR;
            vDistIdx_gpu[iL] = bestDist;
        }  
    }
}




void gpu_stereoMatches(ORB_SLAM3::ORBextractor *mpORBextractorLeft , ORB_SLAM3::ORBextractor *mpORBextractorRight , int time_calls , std::vector<std::vector<size_t>> vRowIndices , std::vector<cv::KeyPoint> mvKeys , 
                      std::vector<cv::KeyPoint> mvKeysRight , 
                      float minZ , float minD , float maxD , int TH_HIGH , int thOrbDist , cv::Mat mDescriptors , cv::Mat mDescriptorsRight ,
                      std::vector<float> mvInvScaleFactors , std::vector<float> mvScaleFactors , std::vector<size_t> size_refer , float mbf , 
                      std::vector<float> &mvDepth_clone , std::vector<float> &mvuRight_clone, std::vector<std::pair<int, int>>& vDistIdx_clone) {

    
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
    int *miniumDist_gpu;
    size_t *miniumDistIndex_gpu;
    const int L = 5;
    const int w = 5;
    int vDistsSize = (2*L+1);  
    float *IL,*IR;
    int *vDistIdx_gpu;
    float *mvDepth_gpu;
    float *mvuRight_gpu;
    int vDistIdx_cpu[N];
    int numBlock,threadForBlock;
    int part_const = ((int)N/nRows) + 1;
    
    // Copia parametri input in memoria costante
    cudaMemcpyToSymbol(minZ_gpu, &minZ, 1 * sizeof(float));
    cudaMemcpyToSymbol(minD_gpu, &minD, 1 * sizeof(float));
    cudaMemcpyToSymbol(maxD_gpu, &maxD, 1 * sizeof(float));
    cudaMemcpyToSymbol(TH_HIGH_gpu, &TH_HIGH, 1 * sizeof(int));
    cudaMemcpyToSymbol(mDescriptors_gpu_cols, &mDescriptors.cols, 1 * sizeof(int));
    cudaMemcpyToSymbol(partition_factor, &part_const, 1 * sizeof(int));
    cudaMemcpyToSymbol(mDescriptors_gpu_lines, &num_elements_left, 1 * sizeof(int));
    cudaMemcpyToSymbol(time_calls_gpu, &time_calls, 1 * sizeof(int));
    cudaMemcpyToSymbol(thOrbDist_gpu, &thOrbDist, 1 * sizeof(int));
    cudaMemcpyToSymbol(Nd, &N, 1 * sizeof(int));
    cudaMemcpyToSymbol(mbf_gpu, &mbf, 1 * sizeof(float));
    cudaMemcpyToSymbol(L_gpu, &L, 1 * sizeof(int));
    cudaMemcpyToSymbol(w_gpu, &w, 1 * sizeof(int));
    

    //Allocazione memoria per array dinamici
    // - Minium Distance
    cudaMalloc(&mvKeys_gpu , sizeof(cv::KeyPoint) * mvKeys.size() );
    cudaMemcpy(mvKeys_gpu, mvKeys.data(), sizeof(cv::KeyPoint) * mvKeys.size(), cudaMemcpyHostToDevice); 
    cudaMalloc(&mvKeysRight_gpu , sizeof(cv::KeyPoint) * mvKeysRight.size() );
    cudaMemcpy(mvKeysRight_gpu, mvKeysRight.data(), sizeof(cv::KeyPoint) * mvKeysRight.size(), cudaMemcpyHostToDevice); 
    cudaMalloc(&mDescriptors_gpu, num_elements_left * sizeof(unsigned char));
    cudaMemcpy(mDescriptors_gpu, (unsigned char*)mDescriptors.data, num_elements_left * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMalloc(&mDescriptorsRight_gpu, num_elements_right * sizeof(unsigned char));
    cudaMemcpy(mDescriptorsRight_gpu, (unsigned char*)mDescriptorsRight.data, num_elements_right * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMalloc(&mvInvScaleFactors_gpu , sizeof(float) * mvInvScaleFactors.size() );   //TODO : cambiare mvInvScaleFactors con mpORBextractorLeft->GetInverseScaleFactors()
    cudaMemcpy(mvInvScaleFactors_gpu, mvInvScaleFactors.data(), sizeof(float) * mvInvScaleFactors.size(), cudaMemcpyHostToDevice);  //TODO : cambiare mvInvScaleFactors con mpORBextractorLeft->GetInverseScaleFactors()
    cudaMalloc(&mvScaleFactors_gpu , sizeof(float) * mvScaleFactors.size() );    //TODO : cambiare mvScaleFactors con mpORBextractorLeft->GetScaleFactors()
    cudaMemcpy(mvScaleFactors_gpu, mvScaleFactors.data(), sizeof(float) * mvScaleFactors.size(), cudaMemcpyHostToDevice); //TODO : cambiare mvScaleFactors con mpORBextractorLeft->GetScaleFactors()
    cudaMalloc(&size_refer_gpu , sizeof(size_t) * size_refer.size() );
    cudaMemcpy(size_refer_gpu, size_refer.data(), sizeof(size_t) * size_refer.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&miniumDist_gpu , sizeof(int) * N);
    cudaMalloc(&miniumDistIndex_gpu , sizeof(size_t) * N);
    // - Sliding window
    cudaMalloc(&IL , sizeof(float) * ((w*2)+1) * ((w*2)+1) * vDistsSize );
    cudaMalloc(&IR , sizeof(float) * ((w*2)+1) * ((w*2)+1) * vDistsSize );
    cudaMalloc(&vDistIdx_gpu , sizeof(int) * N );
    cudaMalloc(&mvDepth_gpu , sizeof(float) * N );
    cudaMalloc(&mvuRight_gpu , sizeof(float) * N );

    
    // vRowIndices is matrix of irregular vector
    // the following code create 2 arrays (1. get n. elements of line(i) 2. get n. elements from 0->i )
    std::vector<size_t> vRowIndices_temp;
    std::vector<size_t> incremental_size_refer;
    incremental_size_refer.resize(size_refer.size());
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
    cudaMalloc(&incremental_size_refer_gpu , sizeof(size_t) * incremental_size_refer.size() );
    cudaMemcpy(incremental_size_refer_gpu, incremental_size_refer.data(), sizeof(size_t) * size_refer.size(), cudaMemcpyHostToDevice); 
    cudaMalloc(&vRowIndices_gpu , sizeof(size_t) * total_element );
    cudaMemcpy(vRowIndices_gpu, vRowIndices_temp.data(), sizeof(size_t) * total_element, cudaMemcpyHostToDevice); 
    

    
    // Minium Distance
    numBlock =nRows;
    threadForBlock = VROWINDICES_MAX_COL;
    printf("\n---------------------------------------{%d}----------------------------------------------\n" , time_calls);
    printf("\nSto per lanciare il test della GPU by Luca Anzaldi: \n");
    printf("\t - Launching function (findMiniumDistance) :  %d block , %d thread for block ---> total = %d threads \n" , numBlock , threadForBlock , numBlock * threadForBlock );
    findMiniumDistance<<<numBlock,threadForBlock>>>(vRowIndices_gpu , mvKeys_gpu , mvKeysRight_gpu , mDescriptors_gpu ,mDescriptorsRight_gpu , mvInvScaleFactors_gpu, mvScaleFactors_gpu , size_refer_gpu , incremental_size_refer_gpu , miniumDist_gpu , miniumDistIndex_gpu );
    cudaDeviceSynchronize();
    
    // Sliding Window
    numBlock =((int)N/NUM_THREAD)+1;
    threadForBlock = NUM_THREAD;
    printf("\t - Launching function (slidingWindow) : %d block , %d thread for block ---> total = %d threads \n" , numBlock , threadForBlock , numBlock * threadForBlock );
    slidingWindow<<< numBlock ,threadForBlock , vDistsSize>>>(mpORBextractorLeft->getRows() , mpORBextractorLeft->getCols() , mpORBextractorLeft->getd_scaleFactor() , mpORBextractorLeft->getd_images(), mpORBextractorLeft->getd_inputImage() , 
                                                                   mpORBextractorRight->getRows() , mpORBextractorRight->getCols() , mpORBextractorRight->getd_scaleFactor(), mpORBextractorRight->getd_images(), mpORBextractorRight->getd_inputImage() , 
                                                                   mvKeys_gpu,mvKeysRight_gpu, mvInvScaleFactors_gpu,mvScaleFactors_gpu,miniumDist_gpu,miniumDistIndex_gpu , IL , IR , vDistIdx_gpu , mvDepth_gpu , mvuRight_gpu);
    cudaDeviceSynchronize();
    
        

    // Fill vDistIdx , mvuRight , mvDepth
    int bestDist;
    cudaMemcpy(vDistIdx_cpu, vDistIdx_gpu, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(mvDepth_clone.data(), mvDepth_gpu, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mvuRight_clone.data(), mvuRight_gpu, N * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0 ; i<N ; i++){
        bestDist = vDistIdx_cpu[i];
        if(bestDist != -1)
            vDistIdx_clone.push_back(std::pair<int,int>(bestDist,i));
    }


    //Memory deallocation
    cudaFree(mvKeys_gpu);
    cudaFree(mvKeysRight_gpu);
    cudaFree(mDescriptors_gpu);
    cudaFree(mDescriptorsRight_gpu);
    cudaFree(mvInvScaleFactors_gpu);
    cudaFree(mvScaleFactors_gpu);
    cudaFree(size_refer_gpu);
    cudaFree(incremental_size_refer_gpu);
    cudaFree(vRowIndices_gpu);
    cudaFree(miniumDist_gpu);
    cudaFree(miniumDistIndex_gpu);
    cudaFree(IL);
    cudaFree(IR);
    cudaFree(vDistIdx_gpu);
    cudaFree(mvDepth_gpu);
    cudaFree(mvuRight_gpu);


}
