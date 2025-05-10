
#ifndef FACESUPER_INTERFACE_HPP
#define FACESUPER_INTERFACE_HPP

#include <iostream>                    // std::cout
#include <fstream>                     // std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector
#include "mydefine.h"

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#ifndef HANDLE
#define	HANDLE		int64_t
#endif // !HANDLE


//. Return	: Handle
//. Initialization function
//. Option : restoration , colorization , inpainting
HANDLE  FaceSuper_Init(int	p_nOpt, torch::Device	p_Device);

//. Return	: 0 if success , other is error code
//. Load value information of model
int  FaceSuper_LoadModel(HANDLE p_h, std::string	p_strModelPath);

//. Return	: 0 if success , other is error code
//. Engine process
int  FaceSuper_EngProc(HANDLE p_h, std::string	p_strInImg, std::string	p_strOutImg);


int  OneFrameOfVideoProc(cv::Mat p_InImg, cv::Mat& p_OutImg);

//. Return	: void
//. Close
void  FaceSuper_Close(HANDLE p_h);

//struct FACESUPER_INFO
//{
//public:
//	bool	        m_nInitOK;
//	int		        m_nOpt;
//	char	        m_szBuff[1024];
//
//
//public:
//	FACESUPER_INFO(int	p_nOpt)
//	{
//		m_nOpt = p_nOpt;
//	};
//};

torch::Device Set_Device();

extern  int         g_nOption;


#endif // FACESUPER_INTERFACE_HPP