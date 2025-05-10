#include <iostream>                    // std::cout
#include <fstream>                     // std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector
#include <random>                      // std::random_device
#include <cstdlib>                     // std::srand, std::rand
// For External Library
#include <torch/torch.h>  
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "FaceSuperEng.hpp"
#include "ImgUtil.h"

// -----------------------------------
// 2. Device Setting Function
// -----------------------------------
torch::Device Set_Device() {

    // (1) GPU Type
    int gpu_id = 0;
    if (torch::cuda::is_available() && gpu_id >= 0) {
        torch::Device device(torch::kCUDA, gpu_id);
        return device;
    }
    // (2) CPU Type
    torch::Device device(torch::kCPU);
    return device;
}
/*
//. main loop
int main()
{
    //.
    wchar_t     w_path[_MAX_PATH] = L"";
    cv::String  w_strInPath = "D:\\aaa.png";;
    cv::String  w_strOutPath = "D:\\bbb.png";;

    cv::Mat img = cv::imread(w_strInPath, cv::IMREAD_COLOR);
    //.
    torch::Device device = Set_Device();

    int         w_nImgDim = 512;
    //. Init model
    FaceSuperEng   w_FaceSuper;

    w_FaceSuper->to(device);
    
    // Get Model
    wcscpy_s(w_path , L"./models/restoration.pth");
    //torch::load(w_FaceSuper, w_path);
    w_FaceSuper->eval();

    // Tensor Forward
//    torch::NoGradGuard no_grad;
//    w_FaceSuper->eval();

    //. 
    torch::Tensor idx_face_tensor = img2tensor(img, true, true);
    //. torchvision
    //normalize(idx_face_tensor, torch::tensor({ 0.5, 0.5, 0.5 }), torch::tensor({ 0.5, 0.5, 0.5 }), true);
    
    idx_face_tensor = idx_face_tensor.unsqueeze(0).to(device);

    try {
        torch::NoGradGuard no_grad;
        torch::Tensor res_tensor = w_FaceSuper->forward(idx_face_tensor, 0.5, true)[0];
        cv::Mat restored_face = tensor2img(res_tensor, true, CV_32F, std::pair<float, float>(-1, 1));
        
        //. del res_tensor
        //. torch.cuda.empty_cache()

        restored_face.convertTo(restored_face, CV_8U);
        cv::imwrite(w_strOutPath, restored_face);
    }
    catch (const std::exception& error) {
        std::cout << "\tFailed restoration: " << error.what() << std::endl;
    }

    return 0;
}
*/