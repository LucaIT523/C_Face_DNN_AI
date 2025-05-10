#pragma once

#include "ClaFaceDetection.h"
#include "opencv2/core.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;

//. Face Detection using opencv, yolov8 onnx
class ClaFaceDetectionOpencv : public ClaFaceDetection
{
public:
	ClaFaceDetectionOpencv();
	~ClaFaceDetectionOpencv();

public:
	//. Model Initialation
	void	InitModel(wchar_t* p_pszModelPath, int	p_nOpt);

	  
	//. Face Detection of Opencv
	//. Return : if	0,	Success. 
	//. 			-1, Error.	
	int		FaceDection(wchar_t* p_pszImagePath, std::vector<ST_FaceRectInfo>& p_stRectInfo);

	int		FaceDection(int* p_pImageBuff, int	p_nW, int	p_nH, std::vector<ST_FaceRectInfo>& p_stRectInfo);

	//. Face Detection of Yolo8
	//. Return : if	0,	Success. 
	//. 			-1, Error.	
	int		FaceDectionYolo(wchar_t* p_pszImagePath, std::vector<ST_FaceRectInfo>& p_stRectInfo);



private:
	//. 



};