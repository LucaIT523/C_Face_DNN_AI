#include <utility>
#include <typeinfo>
#include <cmath>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "VQAutoEncoderEx.hpp"

// Define Namespace
namespace nn = torch::nn;

// ----------------------------------------------------------------------
// struct{VQAutoEncoderExImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
VQAutoEncoderExImpl::VQAutoEncoderExImpl(int imageSize, int cbSize)
{
/*
    std::vector<int> ch_mult = { 1, 2, 2, 4, 4, 8 };


    Encoder         encoder(3, 64, 256, ch_mult, 2, imageSize, 16);
    VectorQuantizer quantize(cbSize, 256, 0.25);
    Generator       generator(64, 256, ch_mult, 2, imageSize,  16);

*/
}


// ----------------------------------------------------------------------
// struct{VQAutoEncoderExImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor VQAutoEncoderExImpl::forward(torch::Tensor x)
{
/*
    torch::Tensor rtn = encoder->forward(x);
    auto [quant, quantStats] = quantize->forward(rtn);
    rtn = generator->forward(quant);
    return { rtn, quantStats, quantStats };
*/
    return x;
}
