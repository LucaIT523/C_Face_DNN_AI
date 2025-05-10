#include <utility>
#include <typeinfo>
#include <cmath>
#include <iostream>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h> 
#include "VQAutoEncoder.hpp"
#include "FaceSuperEng.hpp"


// Define Namespace
namespace nn = torch::nn;
using namespace std;

int         g_nOption ;
//. Calculate mean and std for adaptive_instance_normalization.
//. 
//. Args:
//.    feat(Tensor) : 4D tensor.
//.    eps(float) : A small value added to the variance to avoid
//.    divide - by - zero.Default : 1e-5.
void calc_mean_std(torch::Tensor feat, torch::Tensor& mean, torch::Tensor& std, double eps = 1e-5)
{
    // eps is a small value added to the variance to avoid divide - by - zero.
 //   torch::Tensor size = feat.sizes();
 // 
    // The input feature should be 4D tensor.
    //assert(size.size() == 4);
    int64_t b = feat.size(0);
    int64_t c = feat.size(1);

    //. Calulate
    torch::Tensor feat_var = feat.view({ b, c, -1 }).var(/*dim=*/2) + eps;
    std = feat_var.sqrt().view({ b, c, 1, 1 });
    mean = feat.view({ b, c, -1 }).mean(/*dim=*/2).view({ b, c, 1, 1 });
    //.
    return;
}
//. Adjust the reference features to have the similar color and illuminations
//  as those in the degradate features.
//.  Args:
//      content_feat(Tensor) : The reference feature.
//      style_feat(Tensor) : The degradate features.
torch::Tensor adaptive_instance_normalization(torch::Tensor content_feat, torch::Tensor style_feat)
{
    //. 
    int channels = content_feat.size(0);
    int height = content_feat.size(1);
    int width = content_feat.size(2);
    //size_t sizes_size = content_feat.sizes().size();
    torch::Tensor style_mean, style_std;
    calc_mean_std(style_feat, style_mean, style_std);
    torch::Tensor content_mean, content_std;
    calc_mean_std(content_feat, content_mean, content_std);
    torch::Tensor normalized_feat = (content_feat - content_mean.expand({ content_feat.size(0) , content_feat.size(1) ,content_feat.size(2),content_feat.size(3)})) / content_std.expand({ content_feat.size(0) , content_feat.size(1) ,content_feat.size(2),content_feat.size(3) });
    return normalized_feat * style_std.expand({ content_feat.size(0) , content_feat.size(1) ,content_feat.size(2),content_feat.size(3) }) + style_mean.expand({ content_feat.size(0) , content_feat.size(1) ,content_feat.size(2),content_feat.size(3) });
}
//. This function calculates the mean and standard deviation of a 3D feature tensor.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> _calc_feat_flatten_mean_std(torch::Tensor feat)
{
    //    assert(feat.sizes()[0] == 3);
    assert(feat.dtype() == torch::kFloat32);
    torch::Tensor feat_flatten = feat.view({ 3, -1 });
    torch::Tensor mean = feat_flatten.mean(/*dim=*/-1, /*keepdim=*/true);
    torch::Tensor std = feat_flatten.std(/*dim=*/-1, /*keepdim=*/true);
    return { feat_flatten, mean, std };
}

//. This function calculates the square root of a matrix using singular value decomposition(SVD).
//. SVD decomposes the matrix into three matrices : U, D, and V.The square root of the matrix is 
// then obtained by taking the square root of the diagonal elements of Dand multiplying them with Uand V.
torch::Tensor _mat_sqrt(torch::Tensor x)
{
    auto svd_result = torch::svd(x);
    torch::Tensor U = std::get<0>(svd_result);
    torch::Tensor D = std::get<1>(svd_result);
    torch::Tensor V = std::get<2>(svd_result);
    return torch::mm(torch::mm(U, D.pow(0.5).diag()), V.t());
}
std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}
void FaceSuperEngImpl::Init()
{
    /*
    for (const std::string& module : {"encoder", "quantize", "generator" }) {
        for (auto& param : torch::nn::ModuleDict(*this)[module]->parameters()) {
            param.requires_grad_(false);
        }
    }
*/
    int                      dim = 0;
    int                      cbSize = 0;
    std::vector<std::string> conlist;
    int                      w_conLoop = 0;

    m_bInit = false;
    if (g_nOption == GD_RESTORE_OPT) {
        dim = 512;
        cbSize = 1024;
        conlist = { "32", "64", "128", "256" };
        w_conLoop = 256;
    }
    else if (g_nOption == GD_INPAINT_OPT) {
        dim = 512;
        cbSize = 512;
        conlist = { "32", "64", "128" };
        w_conLoop = 128;

    }
    else if (g_nOption == GD_COLOR_OPT) {
        dim = 512;
        cbSize = 1024;
        conlist = { "32", "64", "128" };
        w_conLoop = 128;
    }
    else {
        return;
    }


    std::vector<int> ch_mult = { 1, 2, 2, 4, 4, 8 };
    layers = 9;
    this->conlist = conlist;
    position_emb = torch::zeros({ 256, dim });
    position_emb = register_parameter("position_emb", position_emb);

    encoder = Encoder(3, 64, 256, ch_mult, 2, 512, 16);
    register_module("encoder", encoder);

    quantize = VectorQuantizer(cbSize, 256, 0.25);
    register_module("quantize", quantize);

    generator = Generator(64, 256, ch_mult, 2, 512, 16);
    register_module("generator", generator);

    feat_emb = torch::nn::Linear(256, dim);
    feat_emb = register_module("feat_emb", feat_emb);


    for (int i = 0; i < layers; ++i) {
        ft_layers->push_back(TransLayer(dim, 2 * dim, 0.0));
    }
    ft_layers = register_module("ft_layers", ft_layers);

    /*
        idx_pred_layer = register_module("idx_pred_layer", torch::nn::Sequential(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({ dim })),
            torch::nn::Linear(dim, cbSize)
        ));
    */
    idx_pred_layer->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({ dim })));
    idx_pred_layer->push_back(torch::nn::Linear((torch::nn::LinearOptions(dim, cbSize).bias(false))));
    idx_pred_layer = register_module("idx_pred_layer", idx_pred_layer);



    channels = { {"16", 512}, {"32", 256}, {"64", 256}, {"128", 128}, {"256", 128}, {"512", 64} };
    fuse_encoder_block = { {"512", 2}, {"256", 5}, {"128", 8}, {"64", 11}, {"32", 14}, {"16", 18} };
    fuse_generator_block = { {"16", 6}, {"32", 9}, {"64", 12}, {"128", 15}, {"256", 18}, {"512", 21} };

    //    fuse_convs_dict = torch::nn::ModuleDict();
    /*
        for (const std::string& k : conlist) {
            int ch = channels[k];
            //std::string     w_regStr = "fuse_convs_dict_" + k;
            //fuse_convs_dict[k] = Fuseblock(ch, ch);
            fuse_convs_dict->insert(k, FuseblockImpl(ch, ch));
        }
    */
    fuse_convs_dict = torch::nn::Sequential();
    for (int i = 0; i < w_conLoop; i++) {
        fuse_convs_dict->push_back(MyTestImpl(32));
        if (i == 31) {
            fuse_convs_dict->push_back(FuseblockImpl(256, 256));
        }
        else if (i == 62) {
            fuse_convs_dict->push_back(FuseblockImpl(256, 256));
        }
        else if (i == 125) {
            fuse_convs_dict->push_back(FuseblockImpl(128, 128));
        }
        else if (i == 252) {
            fuse_convs_dict->push_back(FuseblockImpl(128, 128));
        }
    }
    register_module("fuse_convs_dict", fuse_convs_dict);
    //.OK
    m_bInit = true;
    return;
}
// ----------------------------------------------------------------------
// struct{FaceSuperImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
FaceSuperEngImpl::FaceSuperEngImpl(/*int cbSize, std::vector<std::string> conlist*/)
{
    g_nOption = -1;
}
// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
torch::Tensor FaceSuperEngImpl::Encoder_forward(torch::Tensor p_InTensor, int p_Idx)
{
    torch::Tensor       outTensor;

    if (p_Idx == 0 || p_Idx == 24) {
        outTensor = (encoder->blocks)[p_Idx]->as<torch::nn::Conv2dImpl>()->forward(p_InTensor);
    }
    else if (p_Idx == 1 || p_Idx == 2 || p_Idx == 4 || p_Idx == 5 || p_Idx == 7 || p_Idx == 8 ||
        p_Idx == 10 || p_Idx == 11 || p_Idx == 13 || p_Idx == 14 || p_Idx == 16 || p_Idx == 18  || 
        p_Idx == 20 || p_Idx == 22) {
        outTensor = (encoder->blocks)[p_Idx]->as<ResBlockImpl>()->forward(p_InTensor);
    }
    else if (p_Idx == 3 || p_Idx == 6 || p_Idx == 9 || p_Idx == 12 || p_Idx == 15) {
        outTensor = (encoder->blocks)[p_Idx]->as<DownsampleImpl>()->forward(p_InTensor);
    }
    else if (p_Idx == 17 || p_Idx == 19 || p_Idx == 21) {
        outTensor = (encoder->blocks)[p_Idx]->as<AttnBlockImpl>()->forward(p_InTensor);
    }
    if (p_Idx == 23) {
        outTensor = (encoder->blocks)[p_Idx]->as<torch::nn::GroupNormImpl>()->forward(p_InTensor);
    }

    return outTensor;
}

// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
torch::Tensor FaceSuperEngImpl::Generator_forward(torch::Tensor p_InTensor, int p_Idx)
{
    torch::Tensor       outTensor;

    if (p_Idx == 0 || p_Idx == 24) {
        outTensor = (generator->blocks)[p_Idx]->as<torch::nn::Conv2dImpl>()->forward(p_InTensor);
    }
    else if (p_Idx == 1 || p_Idx == 3 || p_Idx == 4 || p_Idx == 6 || p_Idx == 9 || p_Idx == 10 ||
        p_Idx == 12 || p_Idx == 13 || p_Idx == 15 || p_Idx == 16 || p_Idx == 18 || p_Idx == 19 ||
        p_Idx == 21 || p_Idx == 22) {
        outTensor = (generator->blocks)[p_Idx]->as<ResBlockImpl>()->forward(p_InTensor);
    }
    else if (p_Idx == 8 || p_Idx == 11 || p_Idx == 14 || p_Idx == 17 || p_Idx == 20) {
        outTensor = (generator->blocks)[p_Idx]->as<UpsampleImpl>()->forward(p_InTensor);
    }
    else if (p_Idx == 2 || p_Idx == 5 || p_Idx == 7) {
        outTensor = (generator->blocks)[p_Idx]->as<AttnBlockImpl>()->forward(p_InTensor);
    }
    if (p_Idx == 23) {
        outTensor = (generator->blocks)[p_Idx]->as<torch::nn::GroupNormImpl>()->forward(p_InTensor);
    }

    return outTensor;
}

// ----------------------------------------------------------------------
// struct{FaceSuperImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
int FaceSuperEngImpl::forward(torch::Tensor p_IN, torch::Tensor&  p_OUT, double w_f, bool ada_in)
{

    if (m_bInit == false) {
        return GD_INIT_ERR;
    }

    std::unordered_map<std::string, torch::Tensor> encftdict;
    int     k_num = 0;

    for (int i = 0; i < encoder->blocks->size(); ++i) {

        //. blocks forward
        p_IN = Encoder_forward(p_IN, i);
        //. Save data from conlist
        for (int m = 0; m < conlist.size(); m++) {
            auto w_idxData = fuse_encoder_block.find(conlist[m]);
            if (w_idxData->second == i) {
                encftdict[std::to_string(p_IN.sizes()[3])] = p_IN.clone();
            }
        }

    }
//    std::cout << p_IN.sizes() << std::endl;
//    std::cout << "p_IN[0][0][0] = " << p_IN[0][0][0] << std::endl;

    torch::Tensor rtfeat = p_IN;
    torch::Tensor pos_emb = position_emb.unsqueeze(1).repeat({ 1, p_IN.sizes()[0], 1 });
    torch::Tensor query_emb = feat_emb->forward(rtfeat.flatten(2).permute({ 2, 0, 1 }));


    for (int i = 0; i < ft_layers->size(); ++i) {
        query_emb = ft_layers[i]->as<TransLayerImpl>()->forward(query_emb, pos_emb);
    }

    //std::cout << query_emb.sizes() << std::endl;
    //std::cout << "query_emb[0][0] = " << query_emb[0][0] << std::endl;

    torch::Tensor rtlg_tenso = idx_pred_layer->forward(query_emb).permute({ 1, 0, 2 });
    auto softmax_res = torch::softmax(rtlg_tenso, 2);
    auto topk_res = torch::topk(softmax_res, 1, 2);
    torch::Tensor _idx = std::get<1>(topk_res);

    //std::cout << _idx.sizes() << std::endl;
    //std::cout << "_idx = " << _idx << std::endl;

    torch::Tensor qt_ft = quantize->get_codebook_entry(_idx).detach();

    //std::cout << qt_ft.sizes() << std::endl;
    //std::cout << "qt_ft[0][0][0] = " << qt_ft[0][0][0] << std::endl;


    if (ada_in) {
        qt_ft = adaptive_instance_normalization(qt_ft, rtfeat);
    }

    std::vector<int> fuse_list; //= {9, 12, 15, 18};

    for (const std::string& f_size : conlist) {
        auto w_idxData = fuse_generator_block.find(f_size);
        fuse_list.push_back(w_idxData->second);
    }


    p_OUT = qt_ft;
    for (int i = 0; i < generator->blocks->size(); ++i) {

        //std::cout << p_OUT.sizes() << "  i =  " << i << " " << std::endl;
        //std::cout << "p_OUT[0][0][0] = " << p_OUT[0][0][0] << std::endl;

        p_OUT = Generator_forward(p_OUT, i);

        if (std::find(fuse_list.begin(), fuse_list.end(), i) != fuse_list.end()) {
            std::string k = std::to_string(p_OUT.sizes()[3]);
            k_num = p_OUT.sizes()[3];
            if (w_f > 0) {
                p_OUT = fuse_convs_dict[k_num]->as<FuseblockImpl>()->forward(encftdict[k].detach(), p_OUT, w_f);
            }
        }
    }
 
//    std::cout << rt.sizes() << std::endl;
//    std::cout << "End rt[0][0][0] = " << rt[0][0][0] << std::endl;

    return GD_SUCCESS;
}
// ----------------------------------------------------------------------
// Class {TransLayerImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
TransLayerImpl::TransLayerImpl(int64_t dim, int64_t dimMlp, double dpout)
        : self_attn(torch::nn::MultiheadAttentionOptions(dim, 8).dropout(dpout)),
        linear1(dim, dimMlp),
        linear2(dimMlp, dim),
        norm1(torch::nn::LayerNormOptions({ dim })),
        norm2(torch::nn::LayerNormOptions({ dim })),
        dpout0(torch::nn::DropoutOptions(dpout)),
        dpout1(torch::nn::DropoutOptions(dpout)),
        dpout2(torch::nn::DropoutOptions(dpout)),
        activation(torch::nn::ReLUOptions().inplace(true))
{
    //. Init and register
    register_module("self_attn", self_attn);
    register_module("linear1", linear1);
    register_module("linear2", linear2);
    register_module("norm1", norm1);
    register_module("norm2", norm2);
    register_module("dpout0", dpout0);
    register_module("dpout1", dpout1);
    register_module("dpout2", dpout2);
    register_module("activation", activation);
}
//. This function takes in a tensor and an optional query tensor, 
//  and returns the sum of the two tensors if the query tensor is provided, 
//  otherwise it returns the original tensor.
torch::Tensor TransLayerImpl::pos_emb(torch::Tensor tens, torch::Tensor query) 
{
    return query.defined() ? tens + query : tens;
}
// ----------------------------------------------------------------------
// Class {TransLayerImpl}(nn::Module) -> forward
// ----------------------------------------------------------------------
torch::Tensor TransLayerImpl::forward(torch::Tensor tgt, torch::Tensor query) 
{
    torch::Tensor attn_output;
    torch::Tensor attn_output_weights;

    torch::Tensor qtg = pos_emb(norm1(tgt), query);
    torch::Tensor norm1_res = norm1(tgt);
    std::tie(attn_output, attn_output_weights) = self_attn(qtg, qtg, norm1_res);
    tgt = tgt + dpout1(attn_output);
    torch::Tensor tgt2 = linear2(dpout0(activation(linear1(norm2(tgt)))));
    //torch::Tensor tgt2 = dpout0(norm2(tgt));
    return tgt + dpout2(tgt2);
}

// ----------------------------------------------------------------------
// Class {TransLayerImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
FuseblockImpl::FuseblockImpl(int64_t in_ch, int64_t out_ch)
     : encode_enc(ResBlock(2 * in_ch, out_ch)),
        s_conv1(torch::nn::Conv2dOptions(in_ch, out_ch, 3).padding(1)),
        s_conv2(torch::nn::Conv2dOptions(out_ch, out_ch, 3).padding(1)),
        s_ReLU(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true))),
        scale(torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 3).padding(1)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_ch, out_ch, 3).padding(1)))),
        shift(torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 3).padding(1)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_ch, out_ch, 3).padding(1))))
{
    //. Init and register
    register_module("encode_enc", encode_enc);
    register_module("scale", scale);
    register_module("shift", shift);
//    register_module("s_conv1", s_conv1);
//    register_module("s_conv2", s_conv2);
//    register_module("s_ReLU", s_ReLU);

}
// ----------------------------------------------------------------------
// Class {TransLayerImpl}(nn::Module) -> forward
// ----------------------------------------------------------------------
torch::Tensor FuseblockImpl::forward(torch::Tensor encft, torch::Tensor decft, double w )
{

    encft = encode_enc(torch::cat({ encft, decft }, 1));
    //torch::Tensor scale = m_scale(encft);
    //torch::Tensor w_scale = s_conv2(s_ReLU(s_conv1(encft)));
    //torch::Tensor shift = shift(encft);
    //. Conform ???? (Python code.)
    torch::Tensor scale_data = scale[0]->as<nn::Conv2dImpl>()->forward(encft);
    scale_data = scale[1]->as<nn::LeakyReLUImpl>()->forward(scale_data);
    scale_data = scale[2]->as<nn::Conv2dImpl>()->forward(scale_data);

    torch::Tensor shift_data = shift[0]->as<nn::Conv2dImpl>()->forward(encft);
    shift_data = shift[1]->as<nn::LeakyReLUImpl>()->forward(shift_data);
    shift_data = shift[2]->as<nn::Conv2dImpl>()->forward(shift_data);

    torch::Tensor residual = w * (decft * scale_data + shift_data); /*+ shift_data;*/
    torch::Tensor out = decft + residual;
    return out;
}

