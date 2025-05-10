#ifndef __FFMPEG_UTIL_H
#define __FFMPEG_UTIL_H
//---------------------------------------------------------------------------
#include <iostream>                    // std::cout
#include <fstream>                     // std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector
#include <random>                      // std::random_device
#include <cstdlib>                     // std::srand, std::rand



extern "C" {
#include <libavcodec/avcodec.h>
#ifdef WIN32
#include <libavcodec/codec.h>
#include <libavcodec/packet.h>
#endif
#include <libavformat/avformat.h>
#include <libavutil/timestamp.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
}

#include "FaceSuperInterface.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "mydefine.h"

using namespace std;

#ifndef HANDLE
#define	HANDLE		int64_t
#endif // !HANDLE

#define INBUF_SIZE 4096

int
Extract_Audio_File(std::string	p_strVideoFile_IN, std::string	p_strAudioFile_OUT);

int
Muxer_Video_Audio(std::string	p_strVideoFile_IN, std::string	p_strAudioFile_IN, std::string p_strMuxerFile);

int
Enhance_Video_File(std::string	p_strVideoFile_IN, std::string	p_strVideoFile_OUT);

//int  
//FaceSuper_VideoProc(HANDLE p_h, std::string	p_strInImg, std::string	p_strOutImg);
//
//
//int
//MyEnhance_Video_File(std::string	p_strVideoFile_IN, std::string	p_strVideoFile_OUT);

int  
MyFaceSuper_VideoProc(HANDLE p_h, std::string	p_strInImg, std::string	p_strOutImg);

int
MyFaceSuper_VideoProc_Ex(HANDLE p_h, std::string	p_strInVideo, std::string	p_strOutVideo);


cv::Mat convertFrameToMat(const AVFrame* frame);

AVFrame* cvmatToAvframe(cv::Mat* image, AVFrame* frame);



#endif // __FFMPEG_UTIL_H
