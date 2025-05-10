
#ifndef FACE_RESTORE_UTIL_HPP
#define FACE_RESTORE_UTIL_HPP

#include <iostream>                    // std::cout
#include <fstream>                     // std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector
//. OpenCV
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include<opencv2/ximgproc.hpp>
//.
//#include "mydefine.h"

//using namespace std;

class CFaceRestoreUtil
{

public:
    int                     upscale_factor;
    cv::Size                face_size;
    cv::Mat                 face_template;
//    std::vector<cv::Rect>   det_faces;

    int                     m_nDetFaceCnt;
    std::vector<cv::Mat>    restored_faces;
    std::vector<cv::Mat>    affine_matrices;

    std::vector<cv::Mat>    landmarks;
    std::vector<cv::Mat>    cropped_faces;

    std::vector<cv::Mat>    inverse_affine_matrices;
    cv::Mat                 input_img;
    bool                    checkgray;
    cv::Mat                 m_diffusedImage;
    std::string             m_strImgPath;

    bool                    m_bUse_FaceParsing;

public:
    CFaceRestoreUtil(int upscale_factor, int face_size, double crop_ratio_h, double crop_ratio_w /*, std::string det_model, std::string save_ext, bool template_3points, bool pad_blur, bool use_parse, std::string device*/);

    void read_image(const cv::Mat& img);

    int get_face_landmarks_5(std::string    p_strImgPath, std::string    p_strModelPath, int   resize = -1,  double blur_ratio = 0.01, double eye_dist_threshold = 5.0);

    int get_face_landmarks_5(cv::Mat    p_MatImg, std::string    p_strModelPath, int   resize = -1, double blur_ratio = 0.01, double eye_dist_threshold = 5.0);

    void align_warp_face(/*std::string save_cropped_path, std::string border_mode*/);

    void get_inverse_affine(/*std::string save_inverse_affine_path*/);

    void add_restored_face(cv::Mat restored_face, cv::Mat input_face = cv::Mat());

    cv::Mat paste_orgimage(cv::Mat p_img = cv::Mat(), bool draw_box = false, cv::Mat face_upsampler = cv::Mat());

    void    back_img_diffiusion(cv::Mat    p_MatImg);
};




#endif //. FACE_RESTORE_UTIL_HPP