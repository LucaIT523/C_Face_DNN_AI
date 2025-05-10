//---------------------------------------------------------------------------


#include "ImgUtil.h"
#include <iostream>
#include <fstream>



std::string get_image_type(const cv::Mat& img, bool more_info = true)
{
    std::string r;
    int type = img.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    if (more_info)
        std::cout << "depth: " << img.depth() << " channels: " << img.channels() << std::endl;

    return r;
}

//. 
torch::Tensor img2tensor(cv::Mat img, bool bgr2rgb, bool float32)
{
    torch::Tensor   tensor_img;
    //cv::Mat         img_temp;
    if (img.channels() == 3 && bgr2rgb) {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }

    tensor_img = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, at::kByte);
    tensor_img = tensor_img.permute({ 2, 0, 1 });
    if (float32) {
        tensor_img = tensor_img.to(torch::kFloat).div(255);
    }
    //. 
    return tensor_img;

}

//.

cv::Mat tensor2img(torch::Tensor tensor, bool rgb2bgr /*= true*/, int out_type/* = CV_8U*/, std::pair<float, float> min_max /*= {0, 1}*/)
{

    //std::cout << tensor.sizes() << std::endl;

    //. normalize values
    tensor = tensor.squeeze(0).to(torch::kFloat32).detach().cpu().clamp_(min_max.first, min_max.second);
    tensor = (tensor - min_max.first) / (min_max.second - min_max.first);
    //. Change 
    tensor = tensor.to(torch::kCPU).detach().permute({ 1, 2, 0 }).contiguous();
    tensor = tensor.mul(255.0).clamp(0, 255);

    cv::Mat resultImg(tensor.sizes()[0], tensor.sizes()[1], CV_32FC3);
    memcpy(resultImg.data, tensor.data_ptr(), 4 * sizeof(torch::kFloat32) * tensor.numel());
    cv::cvtColor(resultImg, resultImg, cv::COLOR_RGB2BGR);

    return resultImg;
}

void normalize_custom(torch::Tensor p_Tensor)
{
    std::vector<double> mean = { 0.5, 0.5, 0.5 };
    std::vector<double> std = { 0.5, 0.5, 0.5 };

    int     w_nch = p_Tensor.sizes()[0];

    // Normalize the tensor manually
    for (int channel = 0; channel < w_nch; ++channel) {
        p_Tensor[channel] = p_Tensor[channel].sub_(mean[channel]).div_(std[channel]);
    }

    return;
}




