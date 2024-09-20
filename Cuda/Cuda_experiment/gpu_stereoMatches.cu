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
            //if(iL == 3)
            //printf("GPU {%d} iL = %d inc(%d) element[%d][%d] = %u - %u \n" , time_calls_gpu , iL , incR , countRow , countCol ,V1[index1] , V2[index2]);
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
                               float *scaleFactorsRight, uchar *d_imagesR , uchar *d_inputImageR ,cv::KeyPoint *mvKeys_gpu , cv::KeyPoint *mvKeysRight_gpu , float *mvInvScaleFactors_gpu  , float *mvScaleFactors_gpu , int *miniumDist_gpu , size_t *miniumDistIndex_gpu , float *IL , float *IR , int *tempArray_gpu , float* tempArray_gpu_float){

    extern __shared__ int vDists[];  
    size_t id = threadIdx.x;   
    size_t b_id = blockIdx.x;   
    size_t iL = (b_id * blockDim.x) + id;  // Index is iL of CPU function (iL [0 -> 2026])
    
    int bestDist = INT_MAX;
    int bestincR = 0;    // è il miglior spostamento della windows
    int distN[11]; // TODO  -> Is static could be a problem
    bool exclude = false;
    tempArray_gpu[iL] = -1;
    tempArray_gpu_float[iL] = -1.0;

    if(iL < Nd && miniumDist_gpu[iL] < thOrbDist_gpu ){  //if(bestDist<thOrbDist)

        // coordinates in image pyramid at keypoint scale
        const cv::KeyPoint &kpL = mvKeys_gpu[iL];
        size_t bestIdxR = miniumDistIndex_gpu[iL];
        const float uR0 = mvKeysRight_gpu[bestIdxR].pt.x;        
        const float scaleFactor = mvInvScaleFactors_gpu[kpL.octave];   
        const float scaleduL = round(kpL.pt.x*scaleFactor);     
        const float scaledvL = round(kpL.pt.y*scaleFactor);     
        const float scaleduR0 = round(uR0*scaleFactor);         

        
        // const int w = 5; // save on constant memory GPU

        // The focus now is to transform this ⤋⤋⤋ in a standard arrays
        //cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1); 
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

    
        //printf("{%d}GPU -> iL[%lu] octave = %d row = %d , col = %d , scale factor : %f new_rows = %u , new_cols =%u \n" , time_calls_gpu , index , kpL.octave ,  rows , cols , d_scaleFactor , new_rows , new_cols );

        // Print all pixel of the level
        /*
        int level = kpL.octave;
        int offset_level = (rows * cols) * level;
        uchar *imgPyramid;
        if(iL == 421){
            if(level == 0){
                imgPyramid = d_inputImage;
            }else{
                imgPyramid = d_images;
            }

            for (int i=0 ; i<new_rows ; i++){
                for(int j=0 ; j<new_cols ; j++){
                    int index_of_piramid = ( (i*cols) + j ) + offset_level;
                    printf("{%d}GPU - mvImagePyramid[%d] -  array of size[%d][%d] = [%d][%d] : %u \n" , time_calls_gpu, kpL.octave , rows,cols, i, j, imgPyramid[index_of_piramid]);  
                }
            }       
        }*/
        

        

        // const int L = 5;   // Already on GPU constant_memory
        // vector<float> vDists;  // Save on shared_memory
        // vDists.resize(2*L+1);  // Allocated on sliding_window call

        // calcolano i limiti della finestra scorrevole nella quale verrà effettuata la ricerca dei punti
        const float iniu = scaleduR0+L_gpu-w_gpu;       
        const float endu = scaleduR0+L_gpu+w_gpu+1;

        
        if(iniu<0 || endu >= new_cols_r )   // per evitare di uscire dai range
            exclude = true;
        
        
        for(int incR=-L_gpu , count = 0 ; incR<=+L_gpu && !exclude ; incR++ , count++ ){
            
            //cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
            int i_startIR = scaledvL-w_gpu;
            int i_finalIR = scaledvL+w_gpu+1;
            int j_startIR = scaleduR0+incR-w_gpu;
            int j_finalIR = scaleduR0+incR+w_gpu+1;

            //float dist = cv::norm(IL,IR,cv::NORM_L1);   // Esegue la norma1 tra la finestra_sx e la finestra_dx
            // Sostituto con ⤋⤋⤋

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
            
            /*
            if(iL == 421){
                printf("GPU {%d} RESULT-NORM inc(%d) dist = %d  \n" , time_calls_gpu , incR , dist );
            }
            */

            
            /* FOR GPU - ATTEMPT1
            if(iL == 3){
                int level = kpL.octave;
                int offset_level = (rows * cols) * level;
                int offset_levelR = (rows_r * cols_r) * level;
                uchar *imgPyramid , *imgPyramidRight;
                if(level == 0){
                    imgPyramid = d_inputImage;
                    imgPyramidRight = d_inputImageR;
                }else{
                    imgPyramid = d_images;
                    imgPyramidRight = d_imagesR;
                }
                float norma = 0;
                for (int i=i_startIL , ir = i_startIR  ; i<i_finalIL; i++ , ir++){
                    for(int j=j_startIL , jr = j_startIR; j<j_finalIL ; j++ , jr++){
                        int index = j + (i * new_cols) + offset_level;
                        int indexR = jr + (ir * new_cols_r) + offset_levelR;
                        //printf("GPU {%d} inc(%d) element[%d][%d] = %u - %u \n" , time_calls_gpu ,  incR , i - i_startIL , j - j_startIL ,imgPyramid[index] , imgPyramidRight[indexR]);
                        float f1 = (float) imgPyramid[index];
                        float f2 = (float) imgPyramidRight[indexR];
                        float f3 = abs(f1-f2);
                        printf("GPU {%d} inc(%d) f1,f2,f3 = %f,%f,%f\n" , time_calls_gpu , incR , f1 , f2 , f3);
                    }
                }
                
            }
            */

            /*
            if(iL > 400 && iL < 410){
                printf("GPU {%d} inc(%d) iL = %lu dist = %d\n" , time_calls_gpu , incR , iL ,  dist);
            }*/

            
            if(dist<bestDist)
            {
                bestDist =  dist;
                bestincR = incR;
            }
            
            //vDists[L_gpu+incR] = dist;
            
        }

    }

    

    __syncthreads();
    tempArray_gpu[iL] = bestDist;

    
    
    if(iL < Nd && miniumDist_gpu[iL] < thOrbDist_gpu ){

        // TODO -> this variable are already on the previous if.  you can optimaze
        // coordinates in image pyramid at keypoint scale
        const cv::KeyPoint &kpL = mvKeys_gpu[iL];
        const float &uL = kpL.pt.x;
        size_t bestIdxR = miniumDistIndex_gpu[iL];
        const float uR0 = mvKeysRight_gpu[bestIdxR].pt.x;        
        const float scaleFactor = mvInvScaleFactors_gpu[kpL.octave];      
        const float scaleduR0 = round(uR0*scaleFactor); 

        /*
        if(iL == 3){    
            printf("GPU {%d} iL == 3 BESTDIST = (%d)  \n" , time_calls_gpu , bestDist);  
            /*for(int i=0 ; i<(2*L_gpu+1) ; i++){
                printf("GPU {%d} iL == 3 VDIST[%d] = (%d)  \n" , time_calls_gpu , i ,distN[i] );  
            }*/
        //}
        

        
        /*  // TODO -> DOn't forget this test!!!
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

        
        if(bestincR==-L_gpu || bestincR==L_gpu)
            return;
        
        
        const float dist1 = distN[L_gpu+bestincR-1];
        const float dist2 = distN[L_gpu+bestincR];
        const float dist3 = distN[L_gpu+bestincR+1];  

    
        const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

        tempArray_gpu_float[iL] = deltaR;

        /*  TODO -> ADD CHECK on GPU!!!!
        if(deltaR<-1 || deltaR>1)
            return;
            */

        // Re-scaled coordinate
        /*
        float bestuR = mvScaleFactors_gpu[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

        float disparity = (uL-bestuR);
        */
       
    }
    

}


void test_BestDistAccuracy( int *distanzeMinimeFromGpu , size_t *indici_distanzeMinimeFromGpu , std::vector<int> best_dists , std::vector<size_t> best_dists_index , int time_calls , int N , bool debug  ){

    if(debug){

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
    
    }else{
        
        int cont_valori=0;
        int cont_indici=0;
        for(int i=0 ; i<N ; i++){
            
            if(distanzeMinimeFromGpu[i] == best_dists[i])
                cont_valori++;
            
            if(indici_distanzeMinimeFromGpu[i] == best_dists_index[i])
                cont_indici++;
            
        }
        float percentuale_valori = ((float)cont_valori / (float)N) * 100;
        if(percentuale_valori == 100){
            printf("\t - Test minium distance ✔️ (OK) \n");
        }else{
            printf("\t - Test minium distance - ❌ ( rating = %f)\n" , percentuale_valori);
        }
    }

}


void test_difference_int( int *array1 , std::vector<int> array2 , int n , bool debug , int defaulArrayValue , int testId){

    int cont_valori=0;
    for(int i=0 ; i<n ; i++){
        if(array1[i] == array2[i] || (array2[i] == -1 && array1[i] == defaulArrayValue) )
            cont_valori++;
        else if(debug){
            printf("\t\t❌⇩ ⇩ ⇩  [%d] %d - %d  [ gpu - cpu ] \n" , i , array1[i] , array2[i]);
        }
    }

    float percentuale_valori = ((float)cont_valori / (float) n) * 100;
    if(percentuale_valori == 100){
        printf("\t - Test number %d (OK) ✔️ \n" , testId);
    }else{
        printf("\t - Test number %d - ❌ ( rating = %f)\n" ,  testId , percentuale_valori );
    }
    
}

void test_difference_float( float *array1 , std::vector<float> array2 , int n , bool debug , float defaulArrayValue , int testId ){

    int cont_valori=0;
    for(int i=0 ; i<n ; i++){
        if(array1[i] == array2[i] || (array2[i] == -1 && array1[i] == defaulArrayValue) )
            cont_valori++;
        else if(debug){
            printf("\t\t❌⇩ ⇩ ⇩  [%d] %f - %f \n" , i , array1[i] , array2[i]);
        }
    }

    float percentuale_valori = ((float)cont_valori / (float) n) * 100;
    if(percentuale_valori == 100){
        printf("\t - Test number %d (OK) ✔️ \n" , testId);
    }else{
        printf("\t - Test number %d - ❌ ( rating = %f)\n" , testId ,percentuale_valori);
    }
    
}




void gpu_stereoMatches(ORB_SLAM3::ORBextractor *mpORBextractorLeft , ORB_SLAM3::ORBextractor *mpORBextractorRight , int time_calls , std::vector<std::vector<size_t>> vRowIndices , std::vector<cv::KeyPoint> mvKeys , std::vector<cv::KeyPoint> mvKeysRight , float minZ , float minD , float maxD , int TH_HIGH , int thOrbDist ,cv::Mat mDescriptors , cv::Mat mDescriptorsRight , 
                      std::vector<float> mvInvScaleFactors , std::vector<float> mvScaleFactors , std::vector<size_t> size_refer , std::vector<int> best_dists , std::vector<size_t> best_dists_index ,
                      std::vector<int> bestDist_debug , std::vector<float> dist1_debug , std::vector<float> dist2_debug , std::vector<float> dist3_debug , std::vector<float> deltaR_debug , std::vector<float> bestuR_debug , std::vector<float> disparity_debug, std::vector<float> mvDepth , std::vector<float> mvuRight) {

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
    cudaMemcpyToSymbol(Nd, &N, 1 * sizeof(int));
    

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
    //printf("vrowindices.size() %d\n" , vRowIndices.size() );
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

    /*
    //DEBUG - ToDelete
    for(int i=0 ; i<vRowIndices.size() ; i++){
        printf("%d: %lu \t %lu\n" , i , size_refer[i] , incremental_size_refer[i]);
    }*/
    
    //Test - Functionality of minium distance
    cudaMalloc(&miniumDist_gpu , sizeof(int) * N);
    cudaMalloc(&miniumDistIndex_gpu , sizeof(size_t) * N);

    //Pre-allocation for sliding window
    const int L = 5;
    const int w = 5; //sliding windows_search
    cudaMemcpyToSymbol(L_gpu, &L, 1 * sizeof(int));
    cudaMemcpyToSymbol(w_gpu, &w, 1 * sizeof(int));
    int vDistsSize = (2*L+1);  
    float *IL,*IR;
    cudaMalloc(&IL , sizeof(float) * ((w*2)+1) * ((w*2)+1) * vDistsSize );
    cudaMalloc(&IR , sizeof(float) * ((w*2)+1) * ((w*2)+1) * vDistsSize );

    // DINAMIC ARRAY FOR DEBUGGING (AFTER KERNEL CALL THERE IS AN OTHER)
    int *tempArray_gpu;
    float *tempArray_gpu_float;
    int tempArray_cpu[N];
    float tempArray_cpu_float[N];
    cudaMalloc(&tempArray_gpu , sizeof(int) * N );
    cudaMalloc(&tempArray_gpu_float , sizeof(float) * N );
    


    int numBlock,threadForBlock;
    numBlock =nRows;
    threadForBlock = VROWINDICES_MAX_COL;
    printf("\n---------------------------------------{%d}----------------------------------------------\n" , time_calls);
    printf("\nSto per lanciare il test della GPU by Luca Anzaldi: \n");
    printf("\t - Launching function (findMiniumDistance) :  %d block , %d thread for block ---> total = %d threads \n" , numBlock , threadForBlock , numBlock * threadForBlock );
    findMiniumDistance<<<numBlock,threadForBlock>>>(vRowIndices_gpu , mvKeys_gpu , mvKeysRight_gpu , mDescriptors_gpu ,mDescriptorsRight_gpu , mvInvScaleFactors_gpu, mvScaleFactors_gpu , size_refer_gpu , incremental_size_refer_gpu , miniumDist_gpu , miniumDistIndex_gpu );
    cudaDeviceSynchronize();
    // <<< n. of block , thread for block 
    numBlock =((int)N/NUM_THREAD)+1;
    threadForBlock = NUM_THREAD;
    printf("\t - Launching function (slidingWindow) : %d block , %d thread for block ---> total = %d threads \n" , numBlock , threadForBlock , numBlock * threadForBlock );
    slidingWindow<<< numBlock ,threadForBlock , vDistsSize>>>(mpORBextractorLeft->getRows() , mpORBextractorLeft->getCols() , mpORBextractorLeft->getd_scaleFactor() , mpORBextractorLeft->getd_images(), mpORBextractorLeft->getd_inputImage() , 
                                                                   mpORBextractorRight->getRows() , mpORBextractorRight->getCols() , mpORBextractorRight->getd_scaleFactor(), mpORBextractorRight->getd_images(), mpORBextractorRight->getd_inputImage() , 
                                                                   mvKeys_gpu,mvKeysRight_gpu, mvInvScaleFactors_gpu,mvScaleFactors_gpu,miniumDist_gpu,miniumDistIndex_gpu , IL , IR , tempArray_gpu , tempArray_gpu_float);
    cudaDeviceSynchronize();
    
    // DINAMIC ARRAY FOR DEBUGGING 
    cudaMemcpy(tempArray_cpu, tempArray_gpu, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(tempArray_cpu_float, tempArray_gpu_float, sizeof(float) * N, cudaMemcpyDeviceToHost);
        

    //Test - Test the accuracy of minium distance calculated by Gpu
    //printf("partition_factor :%d , thorbdist : %d \n" , part_const , thOrbDist);
    int distanzeMinime[N];
    size_t distanzeMinimeIndici[N];
    cudaMemcpy(distanzeMinime, miniumDist_gpu, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(distanzeMinimeIndici, miniumDistIndex_gpu, sizeof(size_t) * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    test_BestDistAccuracy(distanzeMinime , distanzeMinimeIndici , best_dists , best_dists_index , time_calls , N , false);

    // Test accuracy 2
    test_difference_int(tempArray_cpu  , bestDist_debug , N , false , INT_MAX , 1);
    test_difference_float(tempArray_cpu_float  , deltaR_debug , N, true , -1.0 , 2);

    
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
