#include <utility>
#include <typeinfo>
#include <cmath>
#include <torch/torch.h>
#include "VQAutoEncoder.hpp"

// Define Namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;

// ----------------------------------------------------------------------
// struct{Downsample}(nn::Module) ->Constructor
// ----------------------------------------------------------------------
DownsampleImpl::DownsampleImpl(int in_channels) {
    conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3).stride(2).padding(0)));
}

// ----------------------------------------------------------------------
// struct{Downsample}(nn::Module) ->Forward
// ----------------------------------------------------------------------
torch::Tensor DownsampleImpl::forward(torch::Tensor x)
{
    std::vector<int64_t> pad = { 0, 1, 0, 1 };
    x = torch::nn::functional::pad(x, pad);
    x = conv->forward(x);
    return x;
}

// ----------------------------------------------------------------------
// struct{Upsample}(nn::Module) ->Constructor
// ----------------------------------------------------------------------
UpsampleImpl::UpsampleImpl(int in_channels)
{
    conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3).stride(1).padding(1)));
}

// ----------------------------------------------------------------------
// struct{Upsample}(nn::Module) ->forward
// ----------------------------------------------------------------------
torch::Tensor UpsampleImpl::forward(torch::Tensor x)
{
    std::vector<int64_t> shape = { x.sizes()[2] * 2 , x.sizes()[3] * 2 };
//    c10::optional <std::vector<double>> new_scale_factor;
//    new_scale_factor.value().push_back= 2.0;
//    x = torch::nn::functional::interpolate(x, torch::nn::functional::InterpolateFuncOptions().scale_factor(new_scale_factor).mode(torch::kNearest));
    x = torch::nn::functional::interpolate(x, torch::nn::functional::InterpolateFuncOptions().size(shape).mode(torch::kNearest));
    x = conv->forward(x);
    return x;
}
// ----------------------------------------------------------------------
// struct{MyTest}(nn::Module) ->Constructor
// ----------------------------------------------------------------------
MyTestImpl::MyTestImpl(int in_channels)
{

}
// ----------------------------------------------------------------------
// struct{MyTest}(nn::Module) ->forward
// ----------------------------------------------------------------------
torch::Tensor MyTestImpl::forward(torch::Tensor x)
{
    torch::Tensor   test;
    return test;
}

// ----------------------------------------------------------------------
// struct{ResBlock}(nn::Module) ->Constructor
// ----------------------------------------------------------------------
ResBlockImpl::ResBlockImpl(int p_ch, int p_out_channels)
{
    chnnel = p_ch;
    if (p_out_channels == 0) {
        out_channels = p_ch;
    }
    else {
        out_channels = p_out_channels;
    }

    norm1 = torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, chnnel).eps(1e-6).affine(true));
    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(chnnel, out_channels, 3).stride(1).padding(1));
    norm2 = torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, out_channels).eps(1e-6).affine(true));
    conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1));

    register_module("norm1", norm1);
    register_module("conv1", conv1);
    register_module("norm2", norm2);
    register_module("conv2", conv2);

    if (chnnel != out_channels) {
        conv_out = torch::nn::Conv2d(torch::nn::Conv2dOptions(chnnel, out_channels, 1).stride(1).padding(0));
        register_module("conv_out", conv_out);
    }

}

// ----------------------------------------------------------------------
// struct{ResBlockImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor ResBlockImpl::forward(torch::Tensor x_in) {
    torch::Tensor x = x_in;
    x = norm1->forward(x);
    x = x * torch::sigmoid(x);
    x = conv1->forward(x);
    x = norm2->forward(x);
    x = x * torch::sigmoid(x);
    x = conv2->forward(x);
    if (chnnel != out_channels) {
        x_in = conv_out->forward(x_in);
    }

    return x + x_in;
}
// ----------------------------------------------------------------------
// struct{AttnBlockImpl}(nn::Module) ->Constructor
// ----------------------------------------------------------------------
AttnBlockImpl::AttnBlockImpl(int   ch)
{
    m_ch = ch;
    norm = torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, m_ch).eps(1e-6).affine(true));
    q = torch::nn::Conv2d(torch::nn::Conv2dOptions(m_ch, m_ch, 1).stride(1).padding(0));
    k = torch::nn::Conv2d(torch::nn::Conv2dOptions(m_ch, m_ch, 1).stride(1).padding(0));
    v = torch::nn::Conv2d(torch::nn::Conv2dOptions(m_ch, m_ch, 1).stride(1).padding(0));
    proj_out = torch::nn::Conv2d(torch::nn::Conv2dOptions(m_ch, m_ch, 1).stride(1).padding(0));

    register_module("norm", norm);
    register_module("q", q);
    register_module("k", k);
    register_module("v", v);
    register_module("proj_out", proj_out);

}

// ----------------------------------------------------------------------
// struct{AttnBlock}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor AttnBlockImpl::forward(torch::Tensor x)
{
    torch::Tensor h_ = x;
    h_ = norm(h_);
    torch::Tensor q_output = q(h_);
    torch::Tensor k_output = k(h_);
    torch::Tensor v_output = v(h_);

    // compute attention
    torch::IntArrayRef q_shape = q_output.sizes();
    int b = q_shape[0];
    int c = q_shape[1];
    int h = q_shape[2];
    int w = q_shape[3];
    torch::Tensor q_reshaped = q_output.reshape({ b, c, h * w }).permute({ 0, 2, 1 });
    torch::Tensor k_reshaped = k_output.reshape({ b, c, h * w });
    torch::Tensor w_ = torch::bmm(q_reshaped, k_reshaped);
    w_ = w_ * pow(c, -0.5);
    w_ = torch::softmax(w_, 2);

    // attend to values
    torch::Tensor v_reshaped = v_output.reshape({ b, c, h * w });
    w_ = w_.permute({ 0, 2, 1 });
    torch::Tensor h__ = torch::bmm(v_reshaped, w_);
    h_ = h__.reshape({ b, c, h, w });

    h_ = proj_out(h_);

    return x + h_;
}

// ----------------------------------------------------------------------
// struct{EncoderImpl}(nn::Module) -> Constructor
// ----------------------------------------------------------------------
EncoderImpl::EncoderImpl(int p_ch, int p_nf, int p_emb_dim, std::vector<int> p_ch_mult, int p_num_res_blocks, int p_resolution, int p_attn_resolutions)
{
    int block_in_ch = 0;
    int block_out_ch = 0;

    nf_ = p_nf;
    num_resolutions_ = p_ch_mult.size();
    num_res_blocks_ = p_num_res_blocks;
    resolution_ = p_resolution;
    attn_resolutions_ = p_attn_resolutions;

    int curr_res = p_resolution;
    std::vector<int> in_ch_mult = { 1 };
    in_ch_mult.insert(in_ch_mult.end(), p_ch_mult.begin(), p_ch_mult.end());

    blocks = torch::nn::Sequential();

    // initial convolution
    blocks->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(p_ch, nf_, 3).stride(1).padding(1)));

    // residual and downsampling blocks, with attention on smaller res (16x16)
    for (int i = 0; i < num_resolutions_; i++) {
        block_in_ch = nf_ * in_ch_mult[i];
        block_out_ch = nf_ * p_ch_mult[i];
        for (int j = 0; j < num_res_blocks_; j++) {
            blocks->push_back(ResBlockImpl(block_in_ch, block_out_ch));
            block_in_ch = block_out_ch;
            if (curr_res == attn_resolutions_) {
                blocks->push_back(AttnBlockImpl(block_in_ch));
            }
        }

        if (i != num_resolutions_ - 1) {
            blocks->push_back(DownsampleImpl(block_in_ch));
            curr_res = curr_res / 2;
        }
    }

    // non-local attention block
    blocks->push_back(ResBlockImpl(block_in_ch, block_in_ch));
    blocks->push_back(AttnBlockImpl(block_in_ch));
    blocks->push_back(ResBlockImpl(block_in_ch, block_in_ch));

    // normalize and convert to latent size
    blocks->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, block_in_ch).eps(1e-6).affine(true)));
    blocks->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(block_in_ch, p_emb_dim, 3).stride(1).padding(1)));

    register_module("blocks", blocks);
}

// ----------------------------------------------------------------------
// struct{EncoderImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor EncoderImpl::forward(torch::Tensor x) {
    return blocks->forward(x);
}


// ----------------------------------------------------------------------
// struct{GeneratorImpl}(nn::Module) -> Constructor
// ----------------------------------------------------------------------
GeneratorImpl::GeneratorImpl(int p_nf, int p_emb_dim, std::vector<int> p_ch_mult, int p_res_blocks, int p_img_size, int p_attn_resolutions)
{
    nf = p_nf;
    ch_mult = p_ch_mult;
    num_resolutions = p_ch_mult.size();
    num_res_blocks = p_res_blocks;
    resolution = p_img_size;
    attn_resolutions = p_attn_resolutions;
    ch = p_emb_dim;
    out_channels = 3;

    int block_in_ch = nf * ch_mult.back();
    int block_out_ch = 0;
    int curr_res = resolution / std::pow(2, num_resolutions - 1);

    blocks = torch::nn::Sequential();
    // Initial conv
    blocks->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(ch, block_in_ch, 3).stride(1).padding(1)));

    // Non-local attention block
    blocks->push_back(ResBlockImpl(block_in_ch, block_in_ch));
    blocks->push_back(AttnBlockImpl(block_in_ch));
    blocks->push_back(ResBlockImpl(block_in_ch, block_in_ch));

    for (int i = num_resolutions - 1; i >= 0; --i) {
        block_out_ch = nf * ch_mult[i];

        for (int j = 0; j < num_res_blocks; ++j) {
            blocks->push_back(ResBlockImpl(block_in_ch, block_out_ch));
            block_in_ch = block_out_ch;

            if (curr_res == attn_resolutions) {
                blocks->push_back(AttnBlockImpl(block_in_ch));
            }
        }

        if (i != 0) {
            blocks->push_back(Upsample(block_in_ch));
            curr_res *= 2;
        }
    }

    blocks->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, block_in_ch).eps(1e-6).affine(true)));
    blocks->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(block_in_ch, out_channels, 3).stride(1).padding(1)));

    this->blocks = register_module("blocks", torch::nn::Sequential(blocks));

}
// ----------------------------------------------------------------------
// struct{GeneratorImpl}(nn::Module) -> forward
// ----------------------------------------------------------------------
torch::Tensor GeneratorImpl::forward(torch::Tensor z)
{
    return blocks->forward(z);

}

// ----------------------------------------------------------------------
// struct{VectorQuantizerImpl}(nn::Module) ->Constructor
// ----------------------------------------------------------------------
VectorQuantizerImpl::VectorQuantizerImpl(const int codebook_size, const size_t emb_dim, float beta)
{
    m_codebook_size = codebook_size;
    m_emb_dim = emb_dim;
    m_beta = beta;
    embedding = torch::nn::Embedding(codebook_size, emb_dim);
    register_module("embedding", embedding);
    //embedding->weight.uniform_(-1.0 / m_codebook_size, 1.0 / m_codebook_size);
}

// ----------------------------------------------------------------------
// struct{VectorQuantizerImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor VectorQuantizerImpl::forward(torch::Tensor z) {

    z = z.permute({ 0, 2, 3, 1 }).contiguous();
    torch::Tensor z_flattened = z.view({ -1, m_emb_dim });

    // distances from z to embeddings e_j(z - e) ^ 2 = z ^ 2 + e ^ 2 - 2 e * z
    torch::Tensor d = (z_flattened.pow(2)).sum(1, true) + (embedding->weight.pow(2)).sum(1) - 2 * torch::matmul(z_flattened, embedding->weight.t());

    torch::Tensor mean_distance = torch::mean(d);
    //. find closest encodings
    torch::Tensor min_encoding_indices = torch::argmin(d, 1).unsqueeze(1);
    //. min_encoding_scores, min_encoding_indices = torch.topk(d, 1, dim = 1, largest = False)
    //. [0-1], higher score, higher confidence
    //. min_encoding_scores = torch.exp(-min_encoding_scores / 10)

    torch::Tensor min_encodings = torch::zeros({ min_encoding_indices.size(0), m_codebook_size }, z.options());
    min_encodings.scatter_(1, min_encoding_indices, 1);

    // get quantized latent vectors
    torch::Tensor z_q = torch::matmul(min_encodings, embedding->weight).view(z.sizes());
    //. compute loss for embedding
    torch::Tensor loss = torch::mean((z_q.detach() - z).pow(2)) + m_beta * torch::mean((z_q - z.detach()).pow(2));
    //. preserve gradients
    z_q = z + (z_q - z).detach();

    //. perplexity
    torch::Tensor e_mean = torch::mean(min_encodings, 0);
    torch::Tensor perplexity = torch::exp(-torch::sum(e_mean * torch::log(e_mean + 1e-10)));


    // reshape back to match original input shape
    z_q = z_q.permute({ 0, 3, 1, 2 }).contiguous();

    /*
                return z_q, loss, {
                    "perplexity": perplexity,
                    "min_encodings" : min_encodings,
                    "min_encoding_indices" : min_encoding_indices,
                    "mean_distance" : mean_distance
    */
    return z;
}
torch::Tensor VectorQuantizerImpl::get_codebook_entry(torch::Tensor indices) 
{
    indices = indices.view({ -1, 1 });
    torch::Tensor min_encodings = torch::zeros({ indices.size(0), m_codebook_size }, indices.options());
    min_encodings.scatter_(1, indices, 1);

    // Get quantized latent vectors
//    auto model_params = this->named_parameters(true);
    torch::Tensor z_q;
    torch::Tensor pair_value;

    for (const auto& pair : this->named_parameters(true)) {
//        std::cout << pair.key() << std::endl;
        pair_value = pair.value();
        z_q = torch::matmul(min_encodings.to(torch::kFloat32), pair_value);
    }

    //if (shape.has_value()) {  // Reshape back to match original input shape
    z_q = z_q.view({ 1, 16 ,16, 256 }).permute({ 0, 3, 1, 2 }).contiguous();
    //}

    return z_q;
}