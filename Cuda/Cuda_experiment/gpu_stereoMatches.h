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




//void gpu_stereoMatches(vector<cv::KeyPoint> mvKeys , float minZ , float minD , float maxD , int TH_HIGH , cv::Mat mDescriptorsRight , vector<float> mvInvScaleFactors , ORBextractor* mpORBextractorLeft , vector<float> mvScaleFactors );
void gpu_stereoMatches(std::vector<cv::KeyPoint> mvKeys , float minZ , float minD , float maxD  );


#endif
