/**
* This file is part of Cuda accelerated ORB-SLAM project by Filippo Muzzini, Nicola Capodieci, Roberto Cavicchioli and Benjamin Rouxel.
 * Implemented by Filippo Muzzini.
 *
 * Based on ORB-SLAM2 (Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós) and ORB-SLAM3 (Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós)
 *
 * Project under GPLv3 Licence
*
*/

#ifndef GPU_STEREOMATCHES
#define GPU_STEREOMATCHES

#include <opencv2/core/hal/interface.h>
#include <vector>
#include <opencv2/features2d.hpp>
#include <ORBextractor.h>



void gpu_stereoMatches(ORB_SLAM3::ORBextractor *mpORBextractorLeft , ORB_SLAM3::ORBextractor *mpORBextractorRight , int time_calls , std::vector<std::vector<size_t>> vRowIndices , std::vector<cv::KeyPoint> mvKeys , std::vector<cv::KeyPoint> mvKeysRight , float minZ , float minD , float maxD , int TH_HIGH , int thOrbDist ,cv::Mat mDescriptors , cv::Mat mDescriptorsRight , 
                      std::vector<float> mvInvScaleFactors , std::vector<float> mvScaleFactors , std::vector<size_t> size_refer , std::vector<int> best_dists , std::vector<size_t> best_dists_index , float mb , float mbf , 
                      std::vector<int> bestDist_debug , std::vector<float> dist1_debug , std::vector<float> dist2_debug , std::vector<float> dist3_debug , std::vector<float> deltaR_debug , std::vector<float> bestuR_debug , std::vector<float> disparity_debug, std::vector<float> &mvDepth_clone , std::vector<float> &mvuRight_clone,
                      std::vector<std::pair<int, int>>& vDistIdx_clone ) ; 


#endif


