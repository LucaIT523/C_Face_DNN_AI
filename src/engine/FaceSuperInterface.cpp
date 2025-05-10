
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "FaceSuperEng.hpp"
#include "FaceParsing.hpp"
#include "LoadModel.h"
#include "ImgUtil.h"
#include "FaceSuperInterface.hpp"
#include "FaceRestoreUtil.hpp"
#include <chrono>
#include "ffmpegUtil.h"

using namespace std::chrono;


FaceSuperEng     g_FaceSuper;
CFaceParsing     g_FaceParsing;
torch::Device	 g_device(torch::kCPU);
std::string      g_ModelPath;

//. Initialization function
HANDLE		
 FaceSuper_Init(
	int		        p_nOpt
,   torch::Device	p_Device
){
    HANDLE              w_h = 0;
    char*               w_sbuff = 0x00;

    if (p_nOpt == GD_RESTORE_OPT || p_nOpt == GD_COLOR_OPT || p_nOpt == GD_INPAINT_OPT) {
        g_nOption = p_nOpt;
    }
    else {
        return w_h;
    }
    w_sbuff = new char[1024];
    memset(w_sbuff, 0x00, sizeof(char) * 1024);

#ifdef WIN32
    if (g_nOption == GD_RESTORE_OPT)
        strcpy_s(w_sbuff, 1024, "\\restore");
    if (g_nOption == GD_COLOR_OPT)
        strcpy_s(w_sbuff, 1024, "\\color");
    if (g_nOption == GD_INPAINT_OPT)
        strcpy_s(w_sbuff, 1024, "\\inpaint");
#else
    if (g_nOption == GD_RESTORE_OPT)
        strcpy(w_sbuff,  "/restore");
    if (g_nOption == GD_COLOR_OPT)
        strcpy(w_sbuff,  "/color");
    if (g_nOption == GD_INPAINT_OPT)
        strcpy(w_sbuff,  "/inpaint");
#endif

    //. OK
//    w_stInfo->m_nInitOK = true;
//    w_stInfo->m_nOpt = g_nOption;
    g_FaceSuper->Init();
    g_FaceSuper->to(p_Device);
    g_FaceParsing->Init();
    g_FaceParsing->to(p_Device);
    w_h = (HANDLE)w_sbuff;
    return w_h;
}

//. Return	: void
//. Close
void
FaceSuper_Close(
    HANDLE p_h
){
    //. Close
    if (p_h != 0) {
        delete[](char*) p_h;
    
    }
    //.
    return;
}

//. Load value information of model
int 
 FaceSuper_LoadModel(
    HANDLE      p_h
,   std::string	p_strModelPath
){
    int             w_nSts = GD_UNKNOWN_ERR;
    char*           w_sbuff = (char*)p_h;
    std::string     w_strPath;


    //. Check Handle
    if (p_h == 0) {
        return GD_INIT_ERR;
    }
    //if (w_stInfo->m_nInitOK != true) {
    //    return GD_INIT_ERR;
    //}

    //. Load Model
    g_ModelPath = p_strModelPath;
    p_strModelPath += w_sbuff;
    w_nSts = SetLoadModelData(g_FaceSuper, p_strModelPath);
    if (w_nSts != 0) {
        w_nSts = GD_LOADMODEL_ERR;
    }
    w_strPath = g_ModelPath;
#ifdef WIN32
    w_strPath += "\\faceparsing";
#else
    w_strPath += "/faceparsing";
#endif
    w_nSts = SetLoadModelData_FaceParsing(g_FaceParsing, w_strPath);
    if (w_nSts != 0) {
        w_nSts = GD_LOADMODEL_ERR;
    }

    //. 
    return w_nSts;
}


//. Engine process
int  FaceSuper_EngProc(
    HANDLE              p_h
,   std::string	        p_strInImg
,   std::string	        p_strOutImg
){
    int             w_nSts = GD_UNKNOWN_ERR;
    char*           w_sbuff = (char*)p_h;
    int             w_nDetFaceCnt = 0;

    //. Check Handle
    if (p_h == 0) {
        return GD_INIT_ERR;
    }
    //if (w_stInfo->m_nInitOK != true) {
    //    return GD_INIT_ERR;
    //}
    //.
    CFaceRestoreUtil        w_CFaceRestoreUtil(2, 512, 1.0, 1.0);

    w_CFaceRestoreUtil.m_bUse_FaceParsing = true;
    {
        auto start = high_resolution_clock::now();
        w_nDetFaceCnt = w_CFaceRestoreUtil.get_face_landmarks_5(p_strInImg, g_ModelPath, 640);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << " get_face_landmarks_5 Time(millisecond)" << duration.count() / 1000.0 << std::endl;

    }
    //. 
    w_CFaceRestoreUtil.align_warp_face();
    //. Face Loop
    for (int k = 0; k < w_CFaceRestoreUtil.cropped_faces.size(); k++) {
        cv::Mat img = w_CFaceRestoreUtil.cropped_faces[k];
        torch::Tensor idx_face_tensor = img2tensor(img, true, true);

        //. torchvision
        normalize_custom(idx_face_tensor);
        idx_face_tensor = idx_face_tensor.unsqueeze(0).to(g_device);


        try {
            torch::NoGradGuard no_grad;
            torch::Tensor res_tensor;

            auto start = high_resolution_clock::now();
            if (g_nOption == GD_RESTORE_OPT) {
                w_nSts = g_FaceSuper->forward(idx_face_tensor, res_tensor, 0.5, true);
            }
            else if (g_nOption == GD_COLOR_OPT) {
                w_nSts = g_FaceSuper->forward(idx_face_tensor, res_tensor, 0, false);
            }
            else if (g_nOption == GD_INPAINT_OPT) {
                //. Mask Init
                torch::Tensor mask = torch::zeros({ 512, 512 });
                torch::Tensor mask_id = idx_face_tensor[0].sum(0);
                mask.masked_fill_(mask_id == 3, 1.0);
                mask = mask.view({ 1, 1, 512, 512 }).to(g_device);
                //. Start Engine
                w_nSts = g_FaceSuper->forward(idx_face_tensor, res_tensor, 1, false);
                //. Mask Process
                res_tensor = (1 - mask) * idx_face_tensor + mask * res_tensor;
            }
            else {
                return w_nSts;
            }
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            std::cout << " FaceSuper_EngProc Time(millisecond)" << duration.count() / 1000.0  << std::endl;
            //. Check result
            if (w_nSts != GD_SUCCESS) {
                return w_nSts;
            }
            cv::Mat restored_face = tensor2img(res_tensor, true, CV_8UC3, { -1, 1 });

            //. del res_tensor
            //. torch.cuda.empty_cache()

            restored_face.convertTo(restored_face, CV_8UC3);

            //std::stringstream w_strOut;
            //w_strOut << "D:\\out_" << k << ".png";
            //cv::imwrite(w_strOut.str().c_str(), restored_face);
            w_CFaceRestoreUtil.add_restored_face(restored_face, img);

            //. OK
            w_nSts = GD_SUCCESS;
        }
        catch (const std::exception& error) {
            w_nSts = GD_UNKNOWN_ERR;
        }

    }

    w_CFaceRestoreUtil.get_inverse_affine();
    cv::Mat savefaceImg = w_CFaceRestoreUtil.paste_orgimage();
    cv::imwrite(p_strOutImg, savefaceImg);
    //. 
    return w_nSts;
}
int  OneFrameOfVideoProc(cv::Mat p_InImg, cv::Mat& p_OutImg) 
{
    int             w_nSts = GD_UNKNOWN_ERR;
    int             w_nDetFaceCnt = 0;

    CFaceRestoreUtil        w_CFaceRestoreUtil(2, 512, 1.0, 1.0);

    w_CFaceRestoreUtil.m_bUse_FaceParsing = true;
    w_nDetFaceCnt = w_CFaceRestoreUtil.get_face_landmarks_5(p_InImg, g_ModelPath, 640);
    //. 
    w_CFaceRestoreUtil.align_warp_face();

    //. Face Loop
    for (int k = 0; k < w_CFaceRestoreUtil.cropped_faces.size(); k++) {
        cv::Mat img = w_CFaceRestoreUtil.cropped_faces[k];
        torch::Tensor idx_face_tensor = img2tensor(img, true, true);
        //. torchvision
        normalize_custom(idx_face_tensor);
        idx_face_tensor = idx_face_tensor.unsqueeze(0).to(g_device);

        try {
            torch::NoGradGuard no_grad;
            torch::Tensor res_tensor;
            w_nSts = g_FaceSuper->forward(idx_face_tensor, res_tensor, 0.5, true);

            //. Check result
            if (w_nSts != GD_SUCCESS) {
                return w_nSts;
            }
            cv::Mat restored_face = tensor2img(res_tensor, true, CV_8UC3, { -1, 1 });

            //. del res_tensor
            //. torch.cuda.empty_cache()
            restored_face.convertTo(restored_face, CV_8UC3);
            w_CFaceRestoreUtil.add_restored_face(restored_face, img);

            //. OK
            w_nSts = GD_SUCCESS;
        }
        catch (const std::exception& error) {
            w_nSts = GD_UNKNOWN_ERR;
        }

    }
    w_CFaceRestoreUtil.get_inverse_affine();
    p_OutImg = w_CFaceRestoreUtil.paste_orgimage();
    return w_nSts;
}
// -----------------------------------
// Device Setting Function
// -----------------------------------
torch::Device Set_Device() {

    // (1) GPU Type
    int gpu_id = 0;
    if (torch::cuda::is_available() && gpu_id >= 0) {
        torch::Device device(torch::kCUDA, gpu_id);
        return device;
    }
    // (2) CPU Type
    torch::Device device(torch::kCPU);
    return device;
}

