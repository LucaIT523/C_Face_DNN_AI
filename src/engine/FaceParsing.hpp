#ifndef __FACE_PARSING_HPP
#define __FACE_PARSING_HPP

#include <iostream>                    // std::cout
#include <fstream>                     // std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector

#include <utility>
#include "torch/script.h"
#include "torch/torch.h"

// Define Namespace
// namespace nn = torch::nn;
using namespace std;

class CFacePsringNormLayerImpl : public torch::nn::Module {
public:
    //.
    CFacePsringNormLayerImpl(int64_t channels, std::string p_norm_type = "bn");
    //.
    torch::Tensor forward(torch::Tensor x);

private:
    std::string             norm_type;
    torch::nn::BatchNorm2d  norm = nullptr;
};

class CFacePsringReluLayerImpl : public torch::nn::Module {
public:
    //.
    CFacePsringReluLayerImpl(int64_t channels, std::string p_relu_type = "relu");
    //.
    torch::Tensor forward(torch::Tensor x);

private:
    std::string             relu_type;
    torch::nn::LeakyReLU    func = nullptr;
};


TORCH_MODULE(CFacePsringNormLayer);
TORCH_MODULE(CFacePsringReluLayer);


class CFacePsringConvLayerImpl : public torch::nn::Module {
public:
    CFacePsringConvLayerImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size = 3, std::string scale = "none",
        std::string p_norm_type = "none", std::string relu_type = "none", bool p_use_pad = true, bool bias = true);

    torch::Tensor forward(torch::Tensor x);

private:
    bool use_pad;
    std::string norm_type;
    std::string scal_type;
    torch::nn::ReflectionPad2d reflection_pad = nullptr;
    torch::nn::Conv2d       conv2d = nullptr;
    CFacePsringReluLayer    relu = nullptr;
    CFacePsringNormLayer    norm = nullptr;
};

TORCH_MODULE(CFacePsringConvLayer);

class CFacePsringShortcut_funcImpl : public torch::nn::Module {
public:
    //.
    CFacePsringShortcut_funcImpl(std::string p_short_type = "short");
    //.
    torch::Tensor forward(torch::Tensor x);

public:
    std::string short_type;
    torch::nn::Conv2d conv{ nullptr };

};
TORCH_MODULE(CFacePsringShortcut_func);

class CFacePsringResidualBlockImpl : public torch::nn::Module {
public:
    //.
    CFacePsringResidualBlockImpl(int c_in, int64_t c_out, std::string relu_type = "prelu", std::string norm_type = "bn", std::string scale = "none");
    //.
    torch::Tensor forward(torch::Tensor x);

public:
    int                             m_short_opt;
    CFacePsringShortcut_func    shortcut_func_Test = nullptr;
    CFacePsringConvLayer        shortcut_func = nullptr;
    CFacePsringConvLayer        conv1 = nullptr;
    CFacePsringConvLayer        conv2 = nullptr;
};

TORCH_MODULE(CFacePsringResidualBlock);

class CFacePsringParseNetImpl : public torch::nn::Module {
public:
    //. 
    CFacePsringParseNetImpl(int64_t in_size = 128, int64_t out_size = 128, int64_t min_feat_size = 32, int64_t base_ch = 64,
        int64_t parsing_ch = 19, int64_t res_depth = 10, std::string relu_type = "leakyrelu", std::string norm_type = "bn",
        std::vector<int> ch_range = { 32, 256 });
 
    //.
    torch::Tensor forward(torch::Tensor x);

private:
    int64_t res_depth;
    torch::nn::Sequential encoder;
    torch::nn::Sequential body;
    torch::nn::Sequential decoder;
    CFacePsringConvLayer out_img_conv = nullptr;
    CFacePsringConvLayer out_mask_conv = nullptr;
};

TORCH_MODULE(CFacePsringParseNet);

class CFaceParsingImpl : public torch::nn::Module 
{
public:
    CFaceParsingImpl() {
        mydata = 0;
    }

    void        Init();

public:
    int                     mydata;
    CFacePsringParseNet     m_CFacePsringParseNet{ nullptr };
};
TORCH_MODULE(CFaceParsing);
#endif // __FACE_PARSING_HPP