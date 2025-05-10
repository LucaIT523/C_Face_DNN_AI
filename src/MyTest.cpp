// MyTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>                    // std::cout
#include <fstream>                     // std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector
#include <cstdio>
#include "FaceSuperEng.hpp"
#include "FaceSuperInterface.hpp"
#include "FaceUtil.h"
#include "ffmpegUtil.h"




// -----------------------------------
// Device Setting Function
// -----------------------------------
std::string  getParamInfo(int argc, char* argv[], std::string p_strKey)
{
    std::string     w_strParam = "";
    for (int i = 0; i < argc; ++i) {
        if (p_strKey == argv[i]) {
            if( i < argc - 1)
                w_strParam = argv[i + 1];
        }
    }
    return w_strParam;
}
bool  checkParamInfo(int argc, char* argv[], std::string p_strKey)
{
    for (int i = 0; i < argc; ++i) {
        if (p_strKey == argv[i]) {
            return true;
        }
    }
    return false;
}

int main(int argc, char* argv[])
{

    // 
    //MyEnhance_Video_File("D:\\demo.mp4", "D:\\temp_video.mp4");
    // Enhance_Video_File("D:\\demo.mp4", "D:\\temp_video.mp4");

//    Muxer_Video_Audio("D:\\demo_1_out.mp4", "D:\\demo_1_audio.aac", "D:\\aaaaaa.mp4");
  //  return 0;

    //. Check Param
    if (argc < 2) {
        std::cout << "Usage : " << std::endl;
        std::cout << "\t -i [input_image_path] -o [output_image_path] -m [model_path] -a [1 | 2 | 3] " << std::endl;
        std::cout << "\t\t (note: -a 1 : restore , 2 : color , 3 : inpaint)" << std::endl;
        std::cout << "\t\t (note: Please add -v option if video file)" << std::endl;
        std::cout << "Example : " << std::endl;
        std::cout << "\tMyTest -i d:\\input\\in.png -o d:\\output\\result.png -m d:\\model -a 1 " << std::endl;
        return 0;
    }

    //.
    int                 w_nSts = GD_UNKNOWN_ERR;
    std::string         w_strInPath = getParamInfo(argc, argv, "-i");
    std::string         w_strOutPath = getParamInfo(argc, argv, "-o");
    std::string         w_strModelFolderPath = getParamInfo(argc, argv, "-m");
    std::string         w_strOption = getParamInfo(argc, argv, "-a");
    bool                w_bVideoOpt = checkParamInfo(argc, argv, "-v");
    int                 w_nOpt = std::stoi(w_strOption);
    HANDLE              w_Handle = 0x00;

    //. 
    w_Handle = FaceSuper_Init(w_nOpt, Set_Device());
    if (w_Handle == 0) {
        std::cout << "FaceSuper_Init Error : " << std::endl;
        return -1;
    }
    else{
        std::cout << "FaceSuper_Init OK : " << std::endl;
    }

    w_nSts = FaceSuper_LoadModel(w_Handle, w_strModelFolderPath);
    if (w_nSts != 0) {
        std::cout << "FaceSuper_LoadModel Error : " << std::endl;
        FaceSuper_Close(w_Handle);
        return -1;
    }
    else {
        std::cout << "FaceSuper_LoadModel OK : " << std::endl;
    }

    if (w_bVideoOpt == true) {
        std::filesystem::path folderPath = std::filesystem::path(w_strInPath).parent_path();
        std::string audio_temp_file = folderPath.string() + "audio_temp.aac";
        std::string video_temp_file = folderPath.string() + "video_temp.mp4";

        //. 
        Extract_Audio_File(w_strInPath, audio_temp_file);

        //. Start engine
//        w_nSts = MyFaceSuper_VideoProc(w_Handle, w_strInPath, video_temp_file);
        MyFaceSuper_VideoProc_Ex(w_Handle, w_strInPath, video_temp_file);

        std::string runpath = std::filesystem::current_path().string();
#ifdef WIN32
        std::string command = runpath + "\\ffmpeg.exe -i " + video_temp_file + " -i " + audio_temp_file + " -c copy " + w_strOutPath;
#else
        std::string command = "ffmpeg -i " + video_temp_file + " -i " + audio_temp_file + " -c copy " + w_strOutPath;
#endif
        system(command.c_str());

        std::remove(audio_temp_file.c_str());
        std::remove(video_temp_file.c_str());

    }
    else {
        w_nSts = FaceSuper_EngProc(w_Handle, w_strInPath, w_strOutPath);
    }
    if (w_nSts != 0) {
        std::cout << "FaceSuper_EngProc Error : " << std::endl;
    }
    else {
        std::cout << "FaceSuper_EngProc OK : " << std::endl;
    }
    //. 
    FaceSuper_Close(w_Handle);

    return GD_SUCCESS;
}

//
//
//int main() {
//    // Register FFmpeg codecs and formats
//    av_register_all();
//
    //// Open the input video file
    //AVFormatContext* inputFormatContext = nullptr;
    //if (avformat_open_input(&inputFormatContext, "input_video.mp4", nullptr, nullptr) != 0) {
    //    std::cerr << "Failed to open input video file" << std::endl;
    //    return -1;
    //}
    //// Retrieve stream information
    //if (avformat_find_stream_info(inputFormatContext, nullptr) < 0) {
    //    std::cerr << "Failed to retrieve input stream information" << std::endl;
    //    avformat_close_input(&inputFormatContext);
    //    return -1;
    //}
    //// Find the video stream
    //int videoStreamIndex = -1;
    //for (unsigned int i = 0; i < inputFormatContext->nb_streams; ++i) {
    //    if (inputFormatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
    //        videoStreamIndex = i;
    //        break;
    //    }
    //}
    //if (videoStreamIndex == -1) {
    //    std::cerr << "Failed to find input video stream" << std::endl;
    //    avformat_close_input(&inputFormatContext);
    //    return -1;
    //}
    //// Get a pointer to the input video codec context
    //AVCodecContext* inputCodecContext = avcodec_alloc_context3(nullptr);
    //if (!inputCodecContext) {
    //    std::cerr << "Failed to allocate input video codec context" << std::endl;
    //    avformat_close_input(&inputFormatContext);
    //    return -1;
    //}
    //if (avcodec_parameters_to_context(inputCodecContext, inputFormatContext->streams[videoStreamIndex]->codecpar) < 0) {
    //    std::cerr << "Failed to copy input codec parameters to video codec context" << std::endl;
    //    avcodec_free_context(&inputCodecContext);
    //    avformat_close_input(&inputFormatContext);
    //    return -1;
    //}
    //// Find the input video decoder
    //AVCodec* inputCodec = avcodec_find_decoder(inputCodecContext->codec_id);
    //if (!inputCodec) {
    //    std::cerr << "Failed to find input video decoder" << std::endl;
    //    avcodec_free_context(&inputCodecContext);
    //    avformat_close_input(&inputFormatContext);
    //    return -1;
    //}
    //// Open the input video codec
    //if (avcodec_open2(inputCodecContext, inputCodec, nullptr) < 0) {
    //    std::cerr << "Failed to open input video codec" << std::endl;
    //    avcodec_free_context(&inputCodecContext);
    //    avformat_close_input(&inputFormatContext);
    //    return -1;
    //}
    //// Create the output video file
    //AVFormatContext* outputFormatContext = nullptr;
    //if (avformat_alloc_output_context2(&outputFormatContext, nullptr, nullptr, "output_video.mp4") < 0) {
    //    std::cerr << "Failed to create output video file" << std::endl;
    //    avcodec_free_context(&inputCodecContext);
    //    avformat_close_input(&inputFormatContext);
    //    return -1;
    //}
    //// Find the output video encoder
    //AVCodec* outputCodec = avcodec_find_encoder(AV_CODEC_ID_H264);
    //if (!outputCodec) {
    //    std::cerr << "Failed to find output video encoder" << std::endl;
    //    avcodec_free_context(&inputCodecContext);
    //    avformat_close_input(&inputFormatContext);
    //    avformat_free_context(outputFormatContext);
    //    return -1;
    //}
    //// Create the output video stream
    //AVStream* outputVideoStream = avformat_new_stream(outputFormatContext, outputCodec);
    //if (!outputVideoStream) {
    //    std::cerr << "Failed to create output video stream" << std::endl;
    //    avcodec_free_context(&inputCodecContext);
    //    avformat_close_input(&inputFormatContext);
    //    avformat_free_context(outputFormatContext);
    //    return -1;
    //}
    //// Copy codec parameters from the input video stream to the output video stream
    //if (avcodec_parameters_copy(outputVideoStream->codecpar, inputFormatContext->streams[videoStreamIndex]->codecpar) < 0) {
    //    std::cerr << "Failed to copy codec parameters from input to output video stream" << std::endl;
    //    avcodec_free_context(&inputCodecContext);
    //    avformat_close_input(&inputFormatContext);
    //    avformat_free_context(outputFormatContext);
    //    return -1;
    //}
    //// Set the output video stream codec
    //outputVideoStream->codecpar->codec_tag = 0;
    //if (avcodec_parameters_to_context(outputVideoStream->codec, outputVideoStream->codecpar) < 0) {
    //    std::cerr << "Failed to copy codec parameters to output video stream codec context" << std::endl;
    //    avcodec_free_context(&inputCodecContext);
    //    avformat_close_input(&inputFormatContext);
    //    avformat_free_context(outputFormatContext);
    //    return -1;
    //}
    //// Open the output video codec
    //if (avcodec_open2(outputVideoStream->codec, outputCodec, nullptr) < 0) {
    //    std::cerr << "Failed to open output video codec" << std::endl;
    //    avcodec_free_context(&inputCodecContext);
    //    avformat_close_input(&inputFormatContext);
    //    avformat_free_context(outputFormatContext);
    //    return -1;
    //}
    //// Allocate video frames
    //AVFrame* inputFrame = av_frame_alloc();
    //AVFrame* outputFrame = av_frame_alloc();
    //if (!inputFrame || !outputFrame) {
    //    std::cerr << "Failed to allocate video frames" << std::endl;
    //    avcodec_free_context(&inputCodecContext);
    //    avformat_close_input(&inputFormatContext);
    //    avcodec_free_context(&outputVideoStream->codec);
    //    avformat_free_context(outputFormatContext);
    //    return -1;
    //}
    //// Determine required buffer size and allocate buffer
    //int numBytes = av_image_get_buffer_size(outputVideoStream->codecpar->format, outputVideoStream->codecpar->width, outputVideoStream->codecpar->height, 1);
    //uint8_t* buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));
    //// Assign appropriate parts of buffer to image planes in outputFrame
    //av_image_fill_arrays(outputFrame->data, outputFrame->linesize, buffer, outputVideoStream->codecpar->format, outputVideoStream->codecpar->width, outputVideoStream->codecpar->height, 1);
    //// Open the output video file for writing
    //if (!(outputFormatContext->oformat->flags & AVFMT_NOFILE)) {
    //    if (avio_open(&outputFormatContext->pb, "output_video.mp4", AVIO_FLAG_WRITE) < 0) {
    //        std::cerr << "Failed to open output video file for writing" << std::endl;
    //        av_frame_free(&inputFrame);
    //        av_frame_free(&outputFrame);
    //        avcodec_free_context(&inputCodecContext);
    //        avformat_close_input(&inputFormatContext);
    //        avcodec_free_context(&outputVideoStream->codec);
    //        avformat_free_context(outputFormatContext);
    //        av_free(buffer);
    //        return -1;
    //    }
    //}
    //// Write the output video file header
    //if (avformat_write_header(outputFormatContext, nullptr) < 0) {
    //    std::cerr << "Failed to write output video file header" << std::endl;
    //    av_frame_free(&inputFrame);
    //    av_frame_free(&outputFrame);
    //    avcodec_free_context(&inputCodecContext);
    //    avformat_close_input(&inputFormatContext);
    //    avcodec_free_context(&outputVideoStream->codec);
    //    avformat_free_context(outputFormatContext);
    //    av_free(buffer);
    //    return -1;
    //}
    //// Read frames from the input video file
    //AVPacket packet;
    //while (av_read_frame(inputFormatContext, &packet) >= 0) {
    //    if (packet.stream_index == videoStreamIndex) {
    //        // Send packet to the input video decoder
    //        if (avcodec_send_packet(inputCodecContext, &packet) < 0) {
    //            std::cerr << "Failed to send packet to input video decoder" << std::endl;
    //            break;
    //        }
    //        // Receive decoded frame from the input video decoder
    //        while (avcodec_receive_frame(inputCodecContext, inputFrame) == 0) {
    //            // Resize the input frame using sws_scale or any other method
    //            // ...
    //            // Encode the resized frame
    //            if (avcodec_send_frame(outputVideoStream->codec, inputFrame) < 0) {
    //                std::cerr << "Failed to send frame to output video encoder" << std::endl;
    //                break;
    //            }
    //            while (avcodec_receive_packet(outputVideoStream->codec, &packet) == 0) {
    //                // Write the encoded packet to the output video file
    //                av_interleaved_write_frame(outputFormatContext, &packet);
    //                av_packet_unref(&packet);
    //            }
    //            av_frame_unref(inputFrame);
    //        }
    //    }
    //    av_packet_unref(&packet);
    //}
    //// Write the output video file trailer
    //av_write_trailer(outputFormatContext);
    //// Free resources
    //av_frame_free(&inputFrame);
    //av_frame_free(&outputFrame);
    //avcodec_free_context(&inputCodecContext);
    //avformat_close_input(&inputFormatContext);
    //avcodec_free_context(&outputVideoStream->codec);
    //avformat_free_context(outputFormatContext);
    //av_free(buffer);
    //return 0;
//}
