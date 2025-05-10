
#ifndef VectorQuantizer_HPP
#define VectorQuantizer_HPP

#include <utility>
#include "torch/script.h"
#include "torch/torch.h"

// Define Namespace
namespace nn = torch::nn;
using namespace std;

// Function Prototype


struct DownsampleImpl : public torch::nn::Module {
public:
    //. Construntor
    DownsampleImpl(int in_channels);
    //. Forward
    torch::Tensor forward(torch::Tensor x);

public:
    torch::nn::Conv2d conv{ nullptr };
};

struct UpsampleImpl : public torch::nn::Module {
public:
    //. Construntor
    UpsampleImpl(int in_channels);
    //. Forward
    torch::Tensor forward(torch::Tensor x);

public:
    torch::nn::Conv2d conv{ nullptr };
};

struct MyTestImpl : public torch::nn::Module {
public:
    //. Construntor
    MyTestImpl(int in_channels);
    //. Forward
    torch::Tensor forward(torch::Tensor x);

public:
    torch::nn::Conv2d conv{ nullptr };
};

// -------------------------------------------------
// struct{ResBlockImpl}(nn::Module)
// -------------------------------------------------
struct ResBlockImpl : nn::Module {
public:
    int             chnnel;         //. in chanels
    int             out_channels;   //. out chanles 
    //. GroupNorm 
    torch::nn::GroupNorm norm1{ nullptr };
    torch::nn::GroupNorm norm2{ nullptr };
    //. Conv2d
    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::Conv2d conv2{ nullptr };
    torch::nn::Conv2d conv_out{ nullptr };


public:
    //. constructor
 //   ResBlockImpl(){}
    ResBlockImpl(int p_ch, int  p_out_channels);
    //. forward
    torch::Tensor forward(torch::Tensor x);
};


// -------------------------------------------------
// struct{AttnBlock}(nn::Module)
// -------------------------------------------------
struct AttnBlockImpl : nn::Module {

public:
    int                     m_ch;   //. chanel
    torch::nn::GroupNorm    norm{ nullptr };
    torch::nn::Conv2d       q{ nullptr };
    torch::nn::Conv2d       k{ nullptr };
    torch::nn::Conv2d       v{ nullptr };
    torch::nn::Conv2d       proj_out{ nullptr };

public:
    //. constructor
 //   AttnBlockImpl() {}
    AttnBlockImpl(int ch);
    //. forward
    torch::Tensor forward(torch::Tensor x);

};


// -------------------------------------------------
// struct{Encoder}(nn::Module)
// -------------------------------------------------
struct EncoderImpl : nn::Module {
public:
    //   bool outermost;
    //   nn::Sequential model;

    int nf_;
    int num_resolutions_;
    int num_res_blocks_;
    int resolution_;
    int attn_resolutions_;

public:
    //. constructor
 //   EncoderImpl() {}
    EncoderImpl(int p_ch, int p_nf, int p_emb_dim, std::vector<int> p_ch_mult, int p_num_res_blocks, int p_resolution, int p_attn_resolutions);
    //. forward
    torch::Tensor forward(torch::Tensor z);
    //. 
    torch::nn::Sequential blocks;

};



// -------------------------------------------------
// struct{GeneratorImpl}(nn::Module)
// -------------------------------------------------
struct GeneratorImpl : nn::Module {
public:
    //    bool outermost;

    int nf;
    std::vector<int> ch_mult;
    int num_resolutions;
    int num_res_blocks;
    int resolution;
    int attn_resolutions;
    int ch;
    int out_channels;


public:
    //. constructor
 //   GeneratorImpl() {}
    GeneratorImpl(int p_nf, int p_emb_dim, std::vector<int> p_ch_mult, int p_res_blocks, int p_img_size, int p_attn_resolutions);
    //. forward
    torch::Tensor forward(torch::Tensor z);
    //torch::nn::Sequential blocks;
    nn::Sequential  blocks;
};

// -------------------------------------------------
// struct{VectorQuantizer}(nn::Module)
// -------------------------------------------------
struct VectorQuantizerImpl : nn::Module {
public:
    //   bool outermost;
    //   nn::Sequential model;
    int m_codebook_size; //.number of embeddings
    int m_emb_dim; //. dimension of embedding
    float m_beta;  //.commitment cost used in loss term, beta* || z_e(x) - sg[e] || ^ 2
    torch::nn::Embedding embedding{ nullptr }; //. embed table

public:
    //. constructor
//    VectorQuantizerImpl() {}
    VectorQuantizerImpl(const int codebook_size, const size_t emb_dim, float beta);
    //. get entry
    torch::Tensor get_codebook_entry(torch::Tensor indices);

    //. forward
    torch::Tensor forward(torch::Tensor z);
};

//. Torch Registpr
TORCH_MODULE(Downsample);
TORCH_MODULE(Upsample);
TORCH_MODULE(MyTest);
TORCH_MODULE(ResBlock);
TORCH_MODULE(AttnBlock);
TORCH_MODULE(Encoder);
TORCH_MODULE(Generator);
TORCH_MODULE(VectorQuantizer);


#endif