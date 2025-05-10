#ifndef FACE_UTIL_H
#define FACE_UTIL_H
//---------------------------------------------------------------------------
#include <iostream>                    // std::cout
#include <fstream>                     // std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector
#include <random>                      // std::random_device
#include <cstdlib>                     // std::srand, std::rand


#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// using namespace std;


// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
std::vector<cv::Point2f>			//. Return - Landmark List.
GetLandmarksOfFace_Dlib(
	cv::Mat			p_Img
,	std::string		p_strModelPath
,	int&			p_nFaceCnt
);

// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
std::vector<cv::Point2f>			//. Return - Landmark List.
GetLandmarksOfFace_Opencv(
	std::string		p_strImgPath
,	int&			p_nFaceCnt
);





#endif // FACE_UTIL_H
