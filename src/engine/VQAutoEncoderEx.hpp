#ifndef VQAutoEncoder_EX_HPP
#define VQAutoEncoder_EX_HPP

#include "VQAutoEncoder.hpp"

// Define Namespace
namespace nn = torch::nn;
using namespace std;


// -------------------------------------------------
// struct{VQAutoEncoderEx}(nn::Module)
// -------------------------------------------------
struct VQAutoEncoderExImpl : nn::Module{
public:



public:
    //. Constructor
 //   VQAutoEncoderExImpl(){}
    VQAutoEncoderExImpl(int imageSize, int cbSize);

    //. forward
    torch::Tensor forward(torch::Tensor x);



};

TORCH_MODULE(VQAutoEncoderEx);

#endif