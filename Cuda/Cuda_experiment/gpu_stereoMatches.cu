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
    size_t b_id = blockIdx.x;   // Each block rappresent one row
    __shared__ int miniumDistShared[MAX_PARTITION_FACTOR]; 
    //__shared__ size_t miniumDistIndexShared[MAX_PARTITION_FACTOR];

    //printf("[GPU] , size_refer_gpu[b_id=%lu]; %lu %lu \n" , b_id ,num_elem_line , size_refer_gpu[b_id]);    //DEBUG 

    //printf("ID(%lu) < num_elem(%lu) of line %lu  mDescript_lines(%d) part_Fact(%d)\n" , id , num_elem_line , b_id , mDescriptors_gpu_lines , partition_factor);

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


__global__ void slidingWindow( int rows , int cols , float *scaleFactors , uchar *d_images ,cv::KeyPoint *mvKeys_gpu , cv::KeyPoint *mvKeysRight_gpu , float *mvInvScaleFactors_gpu  , float *mvScaleFactors_gpu , int *miniumDist_gpu , size_t *miniumDistIndex_gpu){

    size_t id = threadIdx.x;    // Each thread rappresent one element
    size_t b_id = blockIdx.x;   // Each block rappresent one row
    size_t index = (b_id * blockDim.x) + id;  // Index is iL of CPU function (iL [0 -> 2026])

    if(miniumDist_gpu[index] < thOrbDist_gpu ){  //if(bestDist<thOrbDist)

        // coordinates in image pyramid at keypoint scale
        const cv::KeyPoint &kpL = mvKeys_gpu[index];
        size_t bestIdxR = miniumDistIndex_gpu[index];
        const float uR0 = mvKeysRight_gpu[bestIdxR].pt.x;        
        const float scaleFactor = mvInvScaleFactors_gpu[kpL.octave];   
        const float scaleduL = round(kpL.pt.x*scaleFactor);     
        const float scaledvL = round(kpL.pt.y*scaleFactor);     
        const float scaleduR0 = round(uR0*scaleFactor);         

        // sliding window search
        const int w = 5;

        // The focus now is to transform this ⤋⤋⤋ in a standard arrays
        //cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1); 
        int i_start = scaledvL-w;
        int i_final = scaledvL+w+1;
        int j_start = scaleduL-w;
        int j_final = scaleduL+w+1;

        const float d_scaleFactor = scaleFactors[kpL.octave];  //Substitute of { mpORBextractorLeft->mvImagePyramid[kpL.octave] } 
        const uint new_rows = round(rows * 1/d_scaleFactor);
        const uint new_cols = round(cols * 1/d_scaleFactor);

        //printf("{%d}GPU -> iL[%lu] octave = %d row = %d , col = %d , scale factor : %f new_rows = %u , new_cols =%u \n" , time_calls_gpu , index , kpL.octave ,  rows , cols , d_scaleFactor , new_rows , new_cols );

        /*
        //ERROR : Crash application understand why !
        //ERROR : Stampa sempre 0. Capire perchè. d_images è riempita correttamente?
        if(index == 2){
            for (int i=0 ; i<rows ; i++){
                for(int j=0 ; j<cols ; j++){
                    int index_of_piramid = (i*cols) + j;
                    printf("{%d}GPU - mvImagePyramid[%d] -  array of size[%d][%d] = [%d][%d] : %u \n" , time_calls_gpu, kpL.octave , rows,cols,i,j, d_images[index_of_piramid]);  
                }
            }       
        }*/
        

        /*for (int i=i_start ; i<i_final ; i++){
            for(int j=j_start ; j<j_final ; j++){
                printf("%u " , 'a');
            }
            printf("\n");
        }*/


        /*

        CONTINUE FROM HERE...

        int bestDist = INT_MAX;
        int bestincR = 0;    // è il miglior spostamento della windows
        const int L = 5;
        vector<float> vDists;  // ha le distanze tra le finestre nelle immagini sx e dx per ogni possibile spostamento nell'intervallo da -L a +L
        vDists.resize(2*L+1);

        // calcolano i limiti della finestra scorrevole nella quale verrà effettuata la ricerca dei punti
        const float iniu = scaleduR0+L-w;       
        const float endu = scaleduR0+L+w+1;
        if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)   // per evitare di uscire dai range
            continue;

        // Si cerca il migliore incremento e la migliore distanza
        for(int incR=-L; incR<=+L; incR++)
        {
            cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
            float dist = cv::norm(IL,IR,cv::NORM_L1);   // Esegue la norma1 tra la finestra_sx e la finestra_dx
            if(dist<bestDist)
            {
                bestDist =  dist;
                bestincR = incR;
            }

            vDists[L+incR] = dist;
        }

        if(bestincR==-L || bestincR==L)
            continue;

        // Sub-pixel match (Parabola fitting)
        const float dist1 = vDists[L+bestincR-1];
        const float dist2 = vDists[L+bestincR];
        const float dist3 = vDists[L+bestincR+1];

        const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

        if(deltaR<-1 || deltaR>1)
            continue;

        // Re-scaled coordinate
        float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

        float disparity = (uL-bestuR);

        if(disparity>=minD && disparity<maxD)
        {
            if(disparity<=0)
            {
                disparity=0.01;
                bestuR = uL-0.01;
            }
            mvDepth[iL]=mbf/disparity;
            mvuRight[iL] = bestuR;
            vDistIdx.push_back(pair<int,int>(bestDist,iL));
        }
        */
    }

}


void test_BestDistAccuracy( int *distanzeMinimeFromGpu , size_t *indici_distanzeMinimeFromGpu , std::vector<int> best_dists , std::vector<size_t> best_dists_index , int time_calls , int N ){

    printf("{%d}Stampa delle distanze minime e degli indici: \n" , time_calls);
    int cont_valori=0;
    int cont_indici=0;
    for(int i=0 ; i<N ; i++){
        printf("{%d} %d : GPU -> %d[iR=%lu] , CPU -> %d[iR=%lu]" , time_calls ,  i , distanzeMinimeFromGpu[i] , indici_distanzeMinimeFromGpu[i] , best_dists[i] , best_dists_index[i]);
        if(distanzeMinimeFromGpu[i] == best_dists[i])
            cont_valori++;
        else
            printf("  ***");
        if(indici_distanzeMinimeFromGpu[i] == best_dists_index[i])
            cont_indici++;
        else printf("  XXX ");
        
        printf("\n");
    }
    printf("{%d} Percentuale somiglianza valori : %f%% \n" , time_calls , ((float)cont_valori / (float)N) * 100);
    printf("{%d} Percentuale somiglianza indici : %f%% \n\n\n\n" , time_calls , ((float)cont_indici / (float)N) * 100);

}




void gpu_stereoMatches(ORB_SLAM3::ORBextractor *mpORBextractorLeft , ORB_SLAM3::ORBextractor *mpORBextractorRight , int time_calls , std::vector<std::vector<size_t>> vRowIndices , std::vector<cv::KeyPoint> mvKeys , std::vector<cv::KeyPoint> mvKeysRight , float minZ , float minD , float maxD , int TH_HIGH , int thOrbDist ,cv::Mat mDescriptors , cv::Mat mDescriptorsRight , 
                      std::vector<float> mvInvScaleFactors , std::vector<float> mvScaleFactors , std::vector<size_t> size_refer , std::vector<int> best_dists , std::vector<size_t> best_dists_index){

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
    cudaMemcpyToSymbol(thOrbDist_gpu, &thOrbDist, 1 * sizeof(int));
    

    //Allocazione memoria per array dinamici
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


    //TODO -> Evitare di fare questo ciclo e di allocare vRowIndices_temp (spreco di memoria e tempo) OR eseguirlo in GPU
    // vRowIndices is matrix of irregular vector
    // the following code create 2 arrays (1. get n. elements of line(i) 2. get n. elements from 0->i )
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
    cudaMalloc(&incremental_size_refer_gpu , sizeof(size_t) * incremental_size_refer.size() );
    cudaMemcpy(incremental_size_refer_gpu, incremental_size_refer.data(), sizeof(size_t) * size_refer.size(), cudaMemcpyHostToDevice); 
    cudaMalloc(&vRowIndices_gpu , sizeof(size_t) * total_element );
    cudaMemcpy(vRowIndices_gpu, vRowIndices_temp.data(), sizeof(size_t) * total_element, cudaMemcpyHostToDevice); 

    //DEBUG - ToDelete
    for(int i=0 ; i<vRowIndices.size() ; i++){
        printf("%d: %lu \t %lu\n" , i , size_refer[i] , incremental_size_refer[i]);
    }
    
    //Test - Functionality of minium distance
    cudaMalloc(&miniumDist_gpu , sizeof(int) * N);
    cudaMalloc(&miniumDistIndex_gpu , sizeof(size_t) * N);

    printf("Sto per lanciare il test della GPU by Luca Anzaldi: \n");
    findMiniumDistance<<<nRows,VROWINDICES_MAX_COL>>>(vRowIndices_gpu , mvKeys_gpu , mvKeysRight_gpu , mDescriptors_gpu ,mDescriptorsRight_gpu , mvInvScaleFactors_gpu, mvScaleFactors_gpu , size_refer_gpu , incremental_size_refer_gpu , miniumDist_gpu , miniumDistIndex_gpu );
    cudaDeviceSynchronize();
    slidingWindow<<<((int)N/NUM_THREAD),NUM_THREAD>>>(mpORBextractorLeft->getRows() , mpORBextractorLeft->getCols() , mpORBextractorLeft->getd_scaleFactor() , mpORBextractorLeft->getd_images() , mvKeys_gpu,mvKeysRight_gpu,mvInvScaleFactors_gpu,mvScaleFactors_gpu,miniumDist_gpu,miniumDistIndex_gpu);
    cudaDeviceSynchronize();



    //Test - Test the accuracy of minium distance calculated by Gpu
    printf("partition_factor :%d , thorbdist : %d \n" , part_const , thOrbDist);
    int distanzeMinime[N];
    size_t distanzeMinimeIndici[N];
    cudaMemcpy(distanzeMinime, miniumDist_gpu, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(distanzeMinimeIndici, miniumDistIndex_gpu, sizeof(size_t) * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    //test_BestDistAccuracy(distanzeMinime , distanzeMinimeIndici , best_dists , best_dists_index , time_calls , N);

    
    //Memory deallocation
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



