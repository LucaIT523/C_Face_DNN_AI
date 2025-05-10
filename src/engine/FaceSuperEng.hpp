#ifndef __FaceSuper_HPP
#define __FaceSuper_HPP

#include "mydefine.h"
#include "VQAutoEncoder.hpp"
#include "VQAutoEncoderEx.hpp"

//namespace F = torch::nn::Functional;
// Define Namespace
namespace nn = torch::nn;
using namespace std;

//. Transform Layer 
struct TransLayerImpl : public torch::nn::Module {
public:
    //. Constructor
    TransLayerImpl(int64_t dim, int64_t dimMlp, double dpout);

    //.
    torch::Tensor pos_emb(torch::Tensor tens, torch::Tensor query);

    //. forward
    torch::Tensor forward(torch::Tensor tgt, torch::Tensor query);

public:
    //. self-attention mechanism in the model. 
    torch::nn::MultiheadAttention self_attn{ nullptr };
    //.  define class with two linear layers,
    torch::nn::Linear linear1{ nullptr };
    torch::nn::Linear linear2{ nullptr };
    //. define two instances of the LayerNorm class from the nn module. 
    torch::nn::LayerNorm norm1{ nullptr };
    torch::nn::LayerNorm norm2{ nullptr };
    //. define three dropout layers
    torch::nn::Dropout dpout0{ nullptr };
    torch::nn::Dropout dpout1{ nullptr };
    torch::nn::Dropout dpout2{ nullptr };
    //. Set the activation function of a neural network to the GELU function.
    //. self.activation = F.gelu
    //torch::Tensor(*activation)(torch::Tensor);
    torch::nn::ReLU     activation{ nullptr };

};

//. perform fusion operations in a neural network.
struct FuseblockImpl : public torch::nn::Module
{
public:
    //. Constructor
    FuseblockImpl(int64_t in_ch, int64_t out_ch);

    //. forward
    torch::Tensor forward(torch::Tensor encft, torch::Tensor decft, double w = 1.0);

private:

    ResBlock             encode_enc;
    torch::nn::Sequential scale;
    torch::nn::Sequential shift;
    torch::nn::Conv2d       s_conv1{ nullptr };
    torch::nn::Conv2d       s_conv2{ nullptr };
    torch::nn::LeakyReLU    s_ReLU{ nullptr };



};

// -------------------------------------------------
// struct{FaceSuper}(nn::Module)
// -------------------------------------------------
struct FaceSuperEngImpl : public torch::nn::Module 
{
public:
    int                             layers;
    std::vector<std::string>        conlist;
    torch::Tensor                   position_emb;
    torch::nn::Sequential           ft_layers;
    torch::nn::Sequential           idx_pred_layer;
    std::unordered_map<std::string, int> channels;
    std::unordered_map<std::string, int> fuse_encoder_block;
    std::unordered_map<std::string, int> fuse_generator_block;
    torch::nn::Linear               feat_emb{ nullptr };
//    torch::nn::ModuleDict           fuse_convs_dict{ nullptr };
    torch::nn::Sequential           fuse_convs_dict;

    Encoder                         encoder{ nullptr };
    VectorQuantizer                 quantize{ nullptr };
    Generator                       generator{ nullptr };

    bool                            m_bInit;

public:
    //. Constructor
 //   FaceSuperImpl(){}
    FaceSuperEngImpl(/*int   p_nSize, int p_cbSize, std::vector<std::string> p_conlist*/);

    void        Init();

    //. forward
    int    forward(torch::Tensor p_IN, torch::Tensor&  p_OUT, double w_f, bool ada_in);
    //. Encoder Loop
    torch::Tensor   Encoder_forward(torch::Tensor p_InTensor, int p_Idx);
    //. Generator Loop
    torch::Tensor   Generator_forward(torch::Tensor p_InTensor, int p_Idx);
};




TORCH_MODULE(TransLayer);
TORCH_MODULE(Fuseblock);
TORCH_MODULE(FaceSuperEng);



#endif