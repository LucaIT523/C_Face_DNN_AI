#include <utility>
#include <typeinfo>
#include <cmath>
#include "ImgUtil.h"
// For External Library
#include <torch/torch.h>
// For Original Header
#include "FaceParsing.hpp"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Define Namespace
namespace nn = torch::nn;

// ----------------------------------------------------------------------
// struct{CFacePsringNormLayerImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
CFacePsringNormLayerImpl::CFacePsringNormLayerImpl(int64_t channels, std::string p_norm_type /*= "bn"*/)
{
    norm_type = p_norm_type;
    if (norm_type == "bn") {
        norm = register_module("norm", torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(channels).affine(true)));
    }
    else if (norm_type == "none") {
        // Do nothing
    }
}
// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
torch::Tensor CFacePsringNormLayerImpl::forward(torch::Tensor x) {
    if (norm_type == "none") {
        return x;
    }
    else {
        return norm(x);
    }
}
// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
CFacePsringReluLayerImpl::CFacePsringReluLayerImpl(int64_t channels, std::string p_relu_type /*= "relu"*/)
{
    relu_type = p_relu_type;
    if (relu_type == "leakyrelu") {
        func = register_module("func", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
    }
    else if (relu_type == "none") {
        // Do nothing
    }
}
// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
torch::Tensor CFacePsringReluLayerImpl::forward(torch::Tensor x) {
    if (relu_type == "none") {
        return x;
    }
    else {
        return func(x);
    }
}
// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
CFacePsringConvLayerImpl::CFacePsringConvLayerImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size/* = 3*/, std::string scale /*= "none"*/, std::string p_norm_type/* = "none"*/, std::string relu_type /*= "none"*/, bool p_use_pad /*= true*/, bool bias /*= true*/)
{
    norm_type = p_norm_type;
    use_pad = p_use_pad;
    if (norm_type == "bn") {
        bias = false;
    }

    int64_t stride = scale == "down" ? 2 : 1;
    scal_type = scale;

    reflection_pad = torch::nn::ReflectionPad2d(torch::nn::ReflectionPad2dOptions(1));
    conv2d = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(stride).bias(bias));

    relu = CFacePsringReluLayer(out_channels, relu_type);
    norm = CFacePsringNormLayer(out_channels, norm_type);

    register_module("reflection_pad", reflection_pad);
    register_module("conv2d", conv2d);
    register_module("relu", relu);
    register_module("norm", norm);

}
// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
torch::Tensor CFacePsringConvLayerImpl::forward(torch::Tensor x) {
    torch::Tensor out;
    if (scal_type == "up") {
        std::vector<int64_t> shape = { x.sizes()[2] * 2 , x.sizes()[3] * 2 };
        out = torch::nn::functional::interpolate(x, torch::nn::functional::InterpolateFuncOptions().size(shape).mode(torch::kNearest));
    }
    else {
        out = x;
    }

    if (use_pad) {
        out = reflection_pad(out);
    }
    out = conv2d(out);
    out = norm(out);
    out = relu(out);
    return out;
}
// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
CFacePsringShortcut_funcImpl::CFacePsringShortcut_funcImpl(std::string p_short_type /*= "short"*/)
{
    short_type = p_short_type;

}
// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
torch::Tensor CFacePsringShortcut_funcImpl::forward(torch::Tensor x) 
{
    if (short_type == "none") {
        return x;
    }
}
// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
int ch_clip(int p_nch, int min_ch, int max_ch) {
    return std::max(std::min(p_nch, max_ch), min_ch);
}
// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
CFacePsringResidualBlockImpl::CFacePsringResidualBlockImpl(int c_in, int64_t c_out, std::string relu_type /*= "prelu"*/, std::string norm_type/* = "bn"*/, std::string scale/* = "none"*/)
{
    if (scale == "none" && c_in == c_out) {
        shortcut_func_Test = CFacePsringShortcut_func("none");
        m_short_opt = 0;
        register_module("shortcut_func_Test", shortcut_func_Test);

    }
    else {
        shortcut_func = CFacePsringConvLayer(c_in, c_out, 3, scale);
        m_short_opt = 1;
        register_module("shortcut_func", shortcut_func);
    }

    std::unordered_map<std::string, std::vector<std::string>> scale_config_dict = {
        {"down", {"none", "down"}},
        {"up", {"up", "none"}},
        {"none", {"none", "none"}}
    };
    std::vector<std::string> scale_conf = scale_config_dict[scale];

    conv1 = CFacePsringConvLayer(c_in, c_out, 3, scale_conf[0], norm_type, relu_type);
    conv2 = CFacePsringConvLayer(c_out, c_out, 3, scale_conf[1], norm_type, "none");

    register_module("conv1", conv1);
    register_module("conv2", conv2);


}
// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
torch::Tensor CFacePsringResidualBlockImpl::forward(torch::Tensor x) {
    torch::Tensor identity;
    if (m_short_opt == 0) {
        identity = shortcut_func_Test(x);
    }
    else {
        identity = shortcut_func(x);
    }

    torch::Tensor res = conv1(x);
    res = conv2(res);
    return identity + res;
}
// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
CFacePsringParseNetImpl::CFacePsringParseNetImpl(int64_t in_size, int64_t out_size, int64_t min_feat_size, int64_t base_ch, int64_t parsing_ch, int64_t res_depth, std::string relu_type, std::string norm_type, std::vector<int> ch_range)
{
    std::unordered_map<std::string, std::string> act_args = { {"norm_type", norm_type}, {"relu_type", relu_type} };

    int64_t min_ch = ch_range[0];
    int64_t max_ch = ch_range[1];

//    std::cout << min_ch << "   " << max_ch << std::endl;

    min_feat_size = std::min(in_size, min_feat_size);

    int64_t down_steps = std::log2(in_size / min_feat_size);
    int64_t up_steps = std::log2(out_size / min_feat_size);

    encoder = torch::nn::Sequential();
    body = torch::nn::Sequential();
    decoder = torch::nn::Sequential();

    // Define encoder-body-decoder
    encoder->push_back(CFacePsringConvLayer(3, base_ch, 3));
    int64_t head_ch = base_ch;
    for (int64_t i = 0; i < down_steps; ++i) {
        int64_t cin = ch_clip(head_ch, min_ch, max_ch);
        int64_t cout = ch_clip(head_ch * 2, min_ch, max_ch);
        encoder->push_back(CFacePsringResidualBlock(cin, cout, relu_type, norm_type, "down"));
        head_ch = head_ch * 2;
    }

    for (int64_t i = 0; i < res_depth; ++i) {
        body->push_back(CFacePsringResidualBlock(ch_clip(head_ch, min_ch, max_ch), ch_clip(head_ch, min_ch, max_ch), relu_type, norm_type));
    }

    for (int64_t i = 0; i < up_steps; ++i) {
        int64_t cin = ch_clip(head_ch, min_ch, max_ch);
        int64_t cout = ch_clip(head_ch / 2, min_ch, max_ch);
        decoder->push_back(CFacePsringResidualBlock(cin, cout, relu_type, norm_type, "up"));
        head_ch = head_ch / 2;
    }

    out_img_conv = CFacePsringConvLayer(ch_clip(head_ch, min_ch, max_ch), 3);
    out_mask_conv = CFacePsringConvLayer(ch_clip(head_ch, min_ch, max_ch), parsing_ch);

    register_module("encoder", torch::nn::Sequential(encoder));
    register_module("body", torch::nn::Sequential(body));
    register_module("decoder", torch::nn::Sequential(decoder));
    register_module("out_img_conv", out_img_conv);
    register_module("out_mask_conv", out_mask_conv);


}
// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
torch::Tensor CFacePsringParseNetImpl::forward(torch::Tensor x) {
    
    torch::Tensor feat = encoder[0]->as<CFacePsringConvLayerImpl>()->forward(x);
    for (int i = 1; i < encoder->size(); ++i) {
        feat = encoder[i]->as<CFacePsringResidualBlockImpl>()->forward(feat);
    }

    torch::Tensor body_in = feat;
    for (int i = 0; i < body->size(); ++i) {
        body_in= body[i]->as<CFacePsringResidualBlockImpl>()->forward(body_in);
    }

    torch::Tensor out = feat + body_in;
    for (int i = 0; i < decoder->size(); ++i) {
        out = decoder[i]->as<CFacePsringResidualBlockImpl>()->forward(out);
    }

    torch::Tensor out_img = out_img_conv->forward(out);
    torch::Tensor out_mask = out_mask_conv->forward(out);

    //cv::Mat restored_face = tensor2img(out_img, true, CV_8UC3, { -1, 1 });
    //restored_face.convertTo(restored_face, CV_8UC3);
    //cv::imwrite("D:\\test_ooo.png", restored_face);

    return out_mask;
}
// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------

void CFaceParsingImpl::Init()
{
    std::vector<int> ch_rang = { 32, 256 };
//    m_CFacePsringParseNet = CFacePsringParseNet(512, 512, 32, 64, 19, 10, "leakyrelu", "bn", { 32, 256 });
    m_CFacePsringParseNet = CFacePsringParseNet(512, 512, 32, 64, 19, 10, "leakyrelu", "bn", ch_rang);
    register_module("m_CFacePsringParseNet", m_CFacePsringParseNet);
    return;
}