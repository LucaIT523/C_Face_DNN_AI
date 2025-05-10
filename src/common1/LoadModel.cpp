
#include "LoadModel.h"
#include<stdio.h>
#include<cstdlib>
#include<iostream>
#include<string.h>
#include<fstream>
#include <torch/torch.h>
#include <torch/script.h> 


using namespace std;
namespace fs = std::filesystem;

// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
int			            //. Return - 0 if success.
SetLoadModelData(
    FaceSuperEng&       p_FaceSuperEng
,   std::string 		p_strModelFolder)
{

    int                             w_nRtn = -1;
    std::vector<std::string>        w_keyList;
    std::vector<torch::Tensor>      w_TensorList;
    int                             w_nModelFileCnt = GetModelCount(p_strModelFolder);
    std::string                     key_name;

    try {
        torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
        auto model_params = p_FaceSuperEng->named_parameters(true);
        auto model_buffers = p_FaceSuperEng->named_buffers(true);
        
        //for (auto& param : p_FaceSuperEng->named_parameters(true)) {
        //    std::cout << param.key() << ":" << std::endl;
        //}
        //. Get key, Tensor from model file.
        w_nRtn = GetModelInfo(w_keyList, w_TensorList, p_strModelFolder);
        if (w_nRtn != 0) {
            return w_nRtn;
        }
        for (int i = 0; i < w_nModelFileCnt; i++) {
            key_name = w_keyList[i];
            auto* data_ptr = model_params.find(key_name);
            if (data_ptr != nullptr) {
                data_ptr->copy_(w_TensorList[i]);
            }
            else {
                data_ptr = model_buffers.find(key_name);
                if (data_ptr != nullptr) {
                    data_ptr->copy_(w_TensorList[i]);
                }
            }
            //. Test Code
            ///////////////////////////////////////////////////////////////
//            if ("quantize.embedding.weight" == key_name) {
//                std::cout << w_TensorList[i] << std::endl;
//            }
            ///////////////////////////////////////////////////////////////
        }
        torch::autograd::GradMode::set_enabled(true);
    }
    catch (const std::exception& error) {
        std::cout << "SetLoadModelData: " << error.what() << std::endl;
        return -1;
    }
    //. OK
    return 0;
}
// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
int			//. Return - 0 if success.
SetLoadModelData_FaceParsing(
    CFaceParsing&       p_FaceParsing
    , std::string		p_strModelFolder
) {
    int                             w_nRtn = -1;
    std::vector<std::string>        w_keyList;
    std::vector<torch::Tensor>      w_TensorList;
    int                             w_nModelFileCnt = GetModelCount(p_strModelFolder);
    std::string                     key_name;

    try {
        torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
        auto model_params = p_FaceParsing->named_parameters(true);
        auto model_buffers = p_FaceParsing->named_buffers(true);

        //for (auto& param : p_FaceParsing->named_parameters(true)) {
        //    std::cout << param.key() << ":" << std::endl;
        //}
        //. Get key, Tensor from model file.
        w_nRtn = GetModelInfo(w_keyList, w_TensorList, p_strModelFolder);
        if (w_nRtn != 0) {
            return w_nRtn;
        }
        for (int i = 0; i < w_nModelFileCnt; i++) {
            key_name = w_keyList[i];
            key_name = "m_CFacePsringParseNet." + key_name;
            auto* data_ptr = model_params.find(key_name);
            if (data_ptr != nullptr) {
                data_ptr->copy_(w_TensorList[i]);
            }
            else {
                data_ptr = model_buffers.find(key_name);
                if (data_ptr != nullptr) {
                    data_ptr->copy_(w_TensorList[i]);
                }
            }

        }
        torch::autograd::GradMode::set_enabled(true);
    }
    catch (const std::exception& error) {
        std::cout << "\SetLoadModelData_FaceParsing: " << error.what() << std::endl;
        return -1;
    }
    //. OK
    return 0;
}
// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
int			        //. Return - Count.
GetModelCount(
    std::string	p_strModelFolder)
{
    std::size_t number_of_files = 0;
    for (auto const& file : std::filesystem::directory_iterator(p_strModelFolder))
    {
        ++number_of_files;
    }
    return number_of_files;
}

// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
int			        //. Return - 0 if success.	
GetModelInfo(
    std::vector<std::string>&   p_strKeyName
,   std::vector<torch::Tensor>& p_tsData
,   std::string					p_strModelFolder)
{
    try {
        for (auto& file_path : fs::directory_iterator(p_strModelFolder)) {

            std::string     w_filePath = file_path.path().u8string();
            std::string     w_fileName = file_path.path().filename().u8string();
            std::string     w_strKey = w_fileName.substr(0, w_fileName.length() - 4);

            torch::jit::script::Module tensors = torch::jit::load(w_filePath);
            c10::IValue iv = tensors.attr(w_strKey);
            torch::Tensor ts = iv.toTensor();

            std::replace(w_strKey.begin(), w_strKey.end(), '$', '.');
            p_strKeyName.push_back(w_strKey);
            p_tsData.push_back(ts);
        }
    }
    catch (const std::exception& error) {
        std::cout << "\tGetModelInfo: " << error.what() << std::endl;
        return -1;
    }
    return 0;
}
/*
    torch::Tensor ts;
    try
    {
        torch::jit::script::Module tensors = torch::jit::load("restoration.pth");
        c10::IValue iv = tensors.attr("position_emb");
        ts = iv.toTensor();
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the tensor " << std::endl;
        std::cerr << e.msg() << std::endl;
        return 0;
    }

// Model class is inherited from public nn::Module
std::vector<char> Model::get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}


void Model::load_parameters(std::string pt_pth) {
    std::vector<char> f = this->get_the_bytes(pt_pth);
    c10::Dict<IValue, IValue> weights = torch::pickle_load(f).toGenericDict();

    const torch::OrderedDict<std::string, at::Tensor>& model_params = this->named_parameters();
    std::vector<std::string> param_names;
    for (auto const& w : model_params) {
        param_names.push_back(w.key());
    }

    torch::NoGradGuard no_grad;
    for (auto const& w : weights) {
        std::string name = w.key().toStringRef();
        at::Tensor param = w.value().toTensor();

        if (std::find(param_names.begin(), param_names.end(), name) != param_names.end()) {
            model_params.find(name)->copy_(param);
        }
        else {
            std::cout << name << " does not exist among model parameters." << std::endl;
        };

    }
}
*/


/*
    std::string     w_str = "";
    at::Tensor      w_Tensor;
    at::Tensor      w_Tensor_data = torch::zeros({ 256, 512 });

    torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
    //auto new_params = ReadStateDictFromFile(params_path); // implement this
    auto params = w_FaceSuper->named_parameters(true );
    auto buffers = w_FaceSuper->named_buffers(true );
    //for (auto& val : new_params) {
    auto name = "position_emb";//val.key();
    auto* t = params.find(name);
    if (t != nullptr) {
        t->copy_(ts);
    }
    else {
        t = buffers.find(name);
        if (t != nullptr) {
            t->copy_(ts);
        }
    }
    //}
    torch::autograd::GradMode::set_enabled(true);
    const torch::OrderedDict<std::string, at::Tensor>& model_params = w_FaceSuper->named_parameters(false);
    for (const auto& pair : w_FaceSuper->named_parameters()) {
        std::cout << pair.key() << std::endl;
    }

*/

