

#include "FaceUtil.h"
#include "FaceRestoreUtil.hpp"
#include "ImgUtil.h"
#include "FaceParsing.hpp"


extern torch::Device	 g_device;
extern CFaceParsing      g_FaceParsing;
//. Dlib for face detection


//
cv::Mat bgr2gray(cv::Mat img) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat adain_npy(cv::Mat restored_face, cv::Mat input_face) {
    // Color transfer logic goes here
    return restored_face;
}
//
//cv::Rect get_largest_face(std::vector<cv::Rect> det_faces, int h, int w) {
//    auto get_location = [](int val, int length) {
//        if (val < 0)
//            return 0;
//        else if (val > length)
//            return length;
//        else
//            return val;
//    };
//
//    std::vector<int> face_areas;
//    for (const auto& det_face : det_faces) {
//        int left = get_location(det_face.x, w);
//        int right = get_location(det_face.x + det_face.width, w);
//        int top = get_location(det_face.y, h);
//        int bottom = get_location(det_face.y + det_face.height, h);
//        int face_area = (right - left) * (bottom - top);
//        face_areas.push_back(face_area);
//    }
//
//    int largest_idx = std::distance(face_areas.begin(), std::max_element(face_areas.begin(), face_areas.end()));
//    return det_faces[largest_idx];
//}
//
//cv::Rect get_center_face(std::vector<cv::Rect> det_faces, int h, int w, cv::Point center) {
//    if (center == cv::Point(-1, -1))
//        center = cv::Point(w / 2, h / 2);
//
//    std::vector<double> center_dist;
//    for (const auto& det_face : det_faces) {
//        cv::Point face_center((det_face.x + det_face.x + det_face.width) / 2, (det_face.y + det_face.y + det_face.height) / 2);
//        double dist = cv::norm(face_center - center);
//        center_dist.push_back(dist);
//    }
//
//    int center_idx = std::distance(center_dist.begin(), std::min_element(center_dist.begin(), center_dist.end()));
//    return det_faces[center_idx];
//}

CFaceRestoreUtil::CFaceRestoreUtil(int upscale_factor, int face_size, double crop_ratio_h, double crop_ratio_w /*, std::string det_model, std::string save_ext, bool template_3points, bool pad_blur, bool use_parse, std::string device*/)
{
    m_bUse_FaceParsing = false;
    m_nDetFaceCnt = 0;
    checkgray = false;
    this->upscale_factor = upscale_factor;
    this->face_size = cv::Size(face_size * crop_ratio_w, face_size * crop_ratio_h);
    this->face_template = (cv::Mat_<double>(5, 2) << 192.98138, 239.94708, 318.90277, 240.1936, 256.63416, 314.01935, 201.26117, 371.41043, 313.08905, 371.15118);
    this->face_template = this->face_template * (face_size / 512.0);
    if (crop_ratio_h > 1)
        this->face_template.col(1) += face_size * (crop_ratio_h - 1) / 2;
    if (crop_ratio_w > 1)
        this->face_template.col(0) += face_size * (crop_ratio_w - 1) / 2;
    //this->save_ext = save_ext;
    //this->pad_blur = pad_blur;
    //this->checkgray = false;
    //this->g_CFaceUtils = CFaceUtils();
    //this->g_CFaceParsing = CFaceParsing();
}

void CFaceRestoreUtil::read_image(const cv::Mat& img) 
{
    this->input_img = img.clone();

    if (cv::countNonZero(this->input_img > 256) > 0) {
        this->input_img.convertTo(this->input_img, CV_32F, 1.0 / 65535 * 255);
    }

    if (this->input_img.channels() == 1) {
        cv::cvtColor(this->input_img, this->input_img, cv::COLOR_GRAY2BGR);
    }
    //else if (this->input_img.channels() == 4) {
    //    cv::cvtColor(this->input_img, this->input_img, cv::COLOR_BGRA2BGR);
    //}

    //this->checkgray = this->g_CFaceUtils.checkgray(this->input_img, 10);
    //if (this->checkgray)
    //    std::cout << "Grayscale input: True" << std::endl;

    double f = 512.0 / std::min(this->input_img.rows, this->input_img.cols);
    cv::resize(this->input_img, this->input_img, cv::Size(), f, f, cv::INTER_LINEAR);
}

int CFaceRestoreUtil::get_face_landmarks_5(cv::Mat    p_MatImg, std::string    p_strModelPath, int   resize , double blur_ratio , double eye_dist_threshold)
{
    m_strImgPath = "";
    double         scale = 1;
    input_img = p_MatImg;
    if (resize == -1) {
        scale = 1.0;
    }
    else {
 /*       int h = this->input_img.rows;
        int w = this->input_img.cols;
        scale = (double)resize / (double)(std::min(h, w));
        if (scale < 1.0)
            scale = 1.0;
        int new_h = h * scale;
        int new_w = w * scale;

        cv::resize(input_img, input_img, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);*/
    }

    std::vector<cv::Point2f>        w_LandMarksList;
    int                             w_nPos = 0;
    w_LandMarksList = GetLandmarksOfFace_Dlib(input_img, p_strModelPath, m_nDetFaceCnt);
    if (m_nDetFaceCnt <= 0) {
        return 0;
    }
    for (int i = 0; i < m_nDetFaceCnt; i++) {
        w_nPos = i * 5;
        cv::Mat     w_OneFaceLandmarks = (cv::Mat_<double>(5, 2) << w_LandMarksList[w_nPos].x, w_LandMarksList[w_nPos].y, w_LandMarksList[w_nPos + 1].x, w_LandMarksList[w_nPos + 1].y, w_LandMarksList[w_nPos + 2].x, w_LandMarksList[w_nPos + 2].y, w_LandMarksList[w_nPos + 3].x, w_LandMarksList[w_nPos + 3].y, w_LandMarksList[w_nPos + 4].x, w_LandMarksList[w_nPos + 4].y);
        landmarks.push_back(w_OneFaceLandmarks);
    }
    //.
    back_img_diffiusion(p_MatImg);
    //.
    return m_nDetFaceCnt;
}

int CFaceRestoreUtil::get_face_landmarks_5(std::string    p_strImgPath, std::string    p_strModelPath, int   resize, double blur_ratio, double eye_dist_threshold)
{

    //cv::Mat input_img;
    //double scale;
    //if (resize == -1) {
    //    scale = 1;
    //    input_img = this->input_img;
    //}
    //else {
    //    int h = this->input_img.rows;
    //    int w = this->input_img.cols;
    //    scale = resize / std::min(h, w);
    //    scale = std::max(1.0, scale);
    //    int new_h = h * scale;
    //    int new_w = w * scale;
    //    cv::resize(this->input_img, input_img, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);
    //}
    //// Detect faces using the face_detector model
    //std::vector<cv::Rect> bboxes; // Replace with actual face detection logic
    //if(bboxes.empty() || bboxes.size() == 0)
    //    return 0;
    //for (const auto& bbox : bboxes) {
    //    cv::Rect det_face = bbox;
    //    this->det_faces.push_back(det_face);
    //}
    //if (this->det_faces.size() == 0)
    //    return 0;
    //if (only_keep_largest) {
    //    cv::Rect largest_face = get_largest_face(this->det_faces, input_img.rows, input_img.cols);
    //    this->det_faces.clear();
    //    this->det_faces.push_back(largest_face);
    //}
    //else if (only_center_face) {
    //    cv::Rect center_face = get_center_face(this->det_faces, input_img.rows, input_img.cols, cv::Point(-1, -1));
    //    this->det_faces.clear();
    //    this->det_faces.push_back(center_face);
    //}
    double         scale = 1;
    m_strImgPath = p_strImgPath;
    cv::Mat img = cv::imread(p_strImgPath, cv::IMREAD_COLOR);
    input_img = img;
    if (resize == -1) {
        scale = 1.0;
    }
    else {
        //int h = this->input_img.rows;
        //int w = this->input_img.cols;
        //scale = (double)resize / (double)(std::min(h, w));
        //if (scale < 1.0) 
        //    scale = 1.0;
        //int new_h = h * scale;
        //int new_w = w * scale;

        //cv::resize(input_img, input_img, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);
    }

    std::vector<cv::Point2f>        w_LandMarksList;
    int                             w_nPos = 0;
    w_LandMarksList = GetLandmarksOfFace_Dlib(input_img, p_strModelPath, m_nDetFaceCnt);
    if (m_nDetFaceCnt <= 0) {
        return 0;
    }
    for (int i = 0; i < m_nDetFaceCnt; i++) {
        w_nPos = i * 5;
        cv::Mat     w_OneFaceLandmarks = (cv::Mat_<double>(5, 2) << w_LandMarksList[w_nPos].x, w_LandMarksList[w_nPos].y, w_LandMarksList[w_nPos + 1].x, w_LandMarksList[w_nPos + 1].y, w_LandMarksList[w_nPos + 2].x, w_LandMarksList[w_nPos + 2].y, w_LandMarksList[w_nPos + 3].x, w_LandMarksList[w_nPos + 3].y, w_LandMarksList[w_nPos + 4].x, w_LandMarksList[w_nPos + 4].y);
        landmarks.push_back(w_OneFaceLandmarks);
    }
    //.
    back_img_diffiusion(img);
    //.
    return m_nDetFaceCnt;

}

void CFaceRestoreUtil::align_warp_face(/*std::string save_cropped_path, std::string border_mode*/) 
{

    for (int i = 0; i < m_nDetFaceCnt; i++) {
//        std::vector<cv::Point2f> srcPoints = { cv::Point2f(50, 50), cv::Point2f(100, 50), cv::Point2f(50, 100) };
//        std::vector<cv::Point2f> dstPoints = { cv::Point2f(70, 70), cv::Point2f(120, 70), cv::Point2f(70, 120) };
        // Estimate the affine transformation
//        cv::Mat affine_matrix;
 //       cv::Mat affine_matrix = cv::estimateAffinePartial2D(srcPoints, dstPoints);
        cv::Mat affine_matrix = cv::estimateAffinePartial2D(landmarks[i], this->face_template, cv::noArray(), cv::LMEDS);
        this->affine_matrices.push_back(affine_matrix);

        cv::Mat cropped_face;
        cv::warpAffine(this->input_img, cropped_face, affine_matrix, this->face_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(135, 133, 132));
        cropped_faces.push_back(cropped_face);

        //if (!save_cropped_path.empty()) {
        //    std::string path = save_cropped_path.substr(0, save_cropped_path.find_last_of('.'));
        //    std::string save_path = path + "_" + std::to_string(this->affine_matrices.size() - 1) + "." + this->save_ext;
        //    this->g_CFaceUtils.imwrite(cropped_face, save_path);
        //}
    }

    return;
}

void CFaceRestoreUtil::get_inverse_affine(/*std::string save_inverse_affine_path*/) 
{
    std::vector<cv::Mat> inverse_affine_matrices;
    for (const auto& affine_matrix : this->affine_matrices) {
        cv::Mat inverse_affine;
        cv::invertAffineTransform(affine_matrix, inverse_affine);
        inverse_affine *= this->upscale_factor;
        inverse_affine_matrices.push_back(inverse_affine);

        //if (!save_inverse_affine_path.empty()) {
        //    std::string path = save_inverse_affine_path.substr(0, save_inverse_affine_path.find_last_of('.'));
        //    std::string save_path = path + "_" + std::to_string(inverse_affine_matrices.size() - 1) + ".pth";
        //    cv::FileStorage fs(save_path, cv::FileStorage::WRITE);
        //    fs << "inverse_affine" << inverse_affine;
        //    fs.release();
        //}
    }

    this->inverse_affine_matrices = inverse_affine_matrices;
}

void CFaceRestoreUtil::add_restored_face(cv::Mat restored_face, cv::Mat input_face) 
{
    if (this->checkgray) {
        restored_face = bgr2gray(restored_face);
        if (!input_face.empty())
            restored_face = adain_npy(restored_face, input_face);
    }

    this->restored_faces.push_back(restored_face);
}

cv::Mat CFaceRestoreUtil::paste_orgimage(cv::Mat p_img, bool draw_box, cv::Mat face_upsampler)
{
    cv::Mat  test_mask;
    cv::Mat  upsample_img;
//    bool     use_parse = true;

    int     h = this->input_img.rows;
    int     w = this->input_img.cols;
    int     h_up = h * this->upscale_factor;
    int     w_up = w * this->upscale_factor;

    if (m_diffusedImage.empty()) {
        cv::Mat image = cv::imread(m_strImgPath);
        cv::resize(image, upsample_img, cv::Size(w_up, h_up));
    }
    else {
        upsample_img = m_diffusedImage;
    }

    int w_edge = 0;
    int erosion_radius = 0;

    std::vector<cv::Mat> inv_mask_borders;
    for (size_t i = 0; i < this->restored_faces.size(); i++) {
        double extra_offset = (upscale_factor > 1) ? 0.5 * upscale_factor : 0;
        inverse_affine_matrices[i].col(2) += extra_offset;

        cv::Mat inv_restored;
        cv::warpAffine(this->restored_faces[i], inv_restored, inverse_affine_matrices[i], cv::Size(w_up, h_up), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        cv::Mat mask(this->face_size, CV_32F, cv::Scalar(1));
        cv::warpAffine(mask, mask, this->inverse_affine_matrices[i], cv::Size(w_up, h_up), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::Mat inv_mask_erosion;
        cv::erode(mask, inv_mask_erosion, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * this->upscale_factor, 2 * this->upscale_factor)));

        cv::Mat pasted_face;
        cv::merge(std::vector<cv::Mat>{inv_mask_erosion, inv_mask_erosion, inv_mask_erosion}, pasted_face);
        cv::multiply(pasted_face, inv_restored, pasted_face, 1.0, CV_32F);
  
        double total_face_area = cv::sum(inv_mask_erosion)[0];
        w_edge = static_cast<int>(std::sqrt(total_face_area) / 20);
        erosion_radius = 2 * w_edge;

        cv::Mat inv_mask_center;
        cv::erode(inv_mask_erosion, inv_mask_center, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erosion_radius, erosion_radius)));
        cv::Mat inv_soft_mask;
        cv::GaussianBlur(inv_mask_center, inv_soft_mask, cv::Size(erosion_radius + 1 , erosion_radius + 1), 0);

   
        //upsample_img.convertTo(upsample_img, CV_8U);
        //cv::imwrite("D:\\upsample_img_1.png", upsample_img);

        int     MASK_COLORMAP[] = { 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0 };

        //. Face Paring
        if (m_bUse_FaceParsing) {
            cv::Mat face_input;
            cv::resize(this->restored_faces[i], face_input, cv::Size(512, 512), cv::INTER_LINEAR);
          
            torch::Tensor idx_face_tensor = img2tensor(face_input, true, true);
            //. torchvision
            normalize_custom(idx_face_tensor);
            idx_face_tensor = idx_face_tensor.unsqueeze(0).to(g_device);

            torch::Tensor out;
            try {
                torch::NoGradGuard no_grad;
                out = g_FaceParsing->m_CFacePsringParseNet->forward(idx_face_tensor);
            }
            catch (const std::exception& error) {
                std::cout << " m_CFacePsringParseNet Error " << std::endl;
            }
            out = out.argmax(1).squeeze();

           //cv::Mat parse_mask(out.sizes()[0], out.sizes()[1], CV_32F, cv::Scalar(0.0));
           //int     MASK_COLORMAP_SIZE = 19;

           // for (int idx = 0; idx < MASK_COLORMAP_SIZE; idx++) {
           //     for (int i = 0; i < out.sizes()[0]; i++) {
           //         for (int j = 0; j < out.sizes()[1]; j++) {
           //             if (out[i][j].item<int>() == idx) {
           //                 parse_mask.at<float>(i, j) = (float)(MASK_COLORMAP[idx]);
           //             }
           //         }
           //     }
           // }

           cv::Mat parse_mask(out.sizes()[0], out.sizes()[1], CV_32F, cv::Scalar(0.0));
            for (int i = 0; i < out.sizes()[0]; i++) {
                for (int j = 0; j < out.sizes()[1]; j++) {
                    parse_mask.at<float>(i, j) = (float)(MASK_COLORMAP[out[i][j].item<int>()]);
                }
            }
//            cv::GaussianBlur(parse_mask, parse_mask, cv::Size(101, 101), 11);
            cv::GaussianBlur(parse_mask, parse_mask, cv::Size(101, 101), 11);

            const int thres = 10;
            parse_mask.rowRange(0, thres).setTo(0);
            parse_mask.rowRange(parse_mask.rows - thres, parse_mask.rows).setTo(0);
            parse_mask.colRange(0, thres).setTo(0);
            parse_mask.colRange(parse_mask.cols - thres, parse_mask.cols).setTo(0);
//            parse_mask.convertTo(parse_mask, CV_32F, 1.0 / 255.0);
            parse_mask /= 255.0;

            cv::resize(parse_mask, parse_mask, face_size);
            cv::warpAffine(parse_mask, parse_mask, this->inverse_affine_matrices[i], cv::Size(w_up, h_up), 3/*cv::INTER_LINEAR, cv::BORDER_REPLICATE*/);
            cv::Mat inv_soft_parse_mask = parse_mask.clone();


//            inv_soft_mask = parse_mask.clone();


//            cv::Mat inv_soft_parse_mask;
//            cv::merge(std::vector<cv::Mat>{parse_mask, parse_mask, parse_mask}, inv_soft_parse_mask);
            cv::Mat fuse_mask;
            cv::compare(inv_soft_parse_mask, inv_soft_mask, fuse_mask, cv::CMP_LT);
//            fuse_mask.convertTo(fuse_mask, CV_32S);
            fuse_mask = fuse_mask / 255.0;
//            cv::Mat fuse_mask = (inv_soft_parse_mask < inv_soft_mask) / 255;

            //test_mask = parse_mask;
            //test_mask.convertTo(test_mask, CV_8U);
            //cv::imwrite("D:\\parse_mask.png", test_mask);

            //test_mask = inv_soft_parse_mask * 100;
            //test_mask.convertTo(test_mask, CV_8U);
            //cv::imwrite("D:\\inv_soft_parse_mask.png", test_mask);

            //test_mask = fuse_mask * 255;
            //test_mask.convertTo(test_mask, CV_8U);
            //cv::imwrite("D:\\fuse_mask.png", test_mask);

            //test_mask = inv_soft_mask * 100;
            //test_mask.convertTo(test_mask, CV_8U);
            //cv::imwrite("D:\\inv_soft_mask_org_100.png", test_mask);


            cv::Mat sub_fuse_mask = 1 - fuse_mask;

            cv::multiply(inv_soft_parse_mask, fuse_mask, inv_soft_parse_mask, 1.0, CV_32F);
            cv::multiply(inv_soft_mask, sub_fuse_mask, inv_soft_mask, 1.0, CV_32F);
            cv::add(inv_soft_parse_mask, inv_soft_mask, inv_soft_mask, cv::noArray(), CV_32F);
//            inv_soft_mask = inv_soft_parse_mask.mul(fuse_mask) + inv_soft_mask.mul(1 - fuse_mask);

            //test_mask = inv_soft_mask * 100;
            //test_mask.convertTo(test_mask, CV_8U);
            //cv::imwrite("D:\\inv_soft_mask_100.png", test_mask);

        }

        //test_mask = pasted_face;
        //test_mask.convertTo(test_mask, CV_8U);
        //cv::imwrite("D:\\pasted_face.png", test_mask);

        cv::Mat inv_soft_mask_3ch;
        cv::merge(std::vector<cv::Mat>{inv_soft_mask, inv_soft_mask, inv_soft_mask}, inv_soft_mask_3ch);

        cv::Mat an_inv_soft_mask_3ch;
        cv::merge(std::vector<cv::Mat>{1- inv_soft_mask, 1 - inv_soft_mask, 1 - inv_soft_mask}, an_inv_soft_mask_3ch);

        cv::Mat pasted_face_make_result;
        cv::multiply(inv_soft_mask_3ch, pasted_face, pasted_face_make_result, 1.0, CV_32F);

        ////////////////////////////////////////////////////////////////////////////
        //pasted_face.convertTo(pasted_face, CV_8U);
        //cv::imwrite("D:\\pasted_face.png", pasted_face);

        //pasted_face_make_result.convertTo(pasted_face_make_result, CV_8U);
        //cv::imwrite("D:\\pasted_face_make_result.png", pasted_face_make_result);

        //inv_soft_mask_3ch.convertTo(inv_soft_mask_3ch, CV_8U);
        //cv::imwrite("D:\\inv_soft_mask_3ch.png", inv_soft_mask_3ch);

        //an_inv_soft_mask_3ch.convertTo(an_inv_soft_mask_3ch, CV_8U);
        //cv::imwrite("D:\\an_inv_soft_mask_3ch.png", an_inv_soft_mask_3ch);
        //////////////////////////////////////////////////////////////////////
        cv::Mat inver;
        cv::multiply(an_inv_soft_mask_3ch, upsample_img, inver, 1.0, CV_32F);

        //inver.convertTo(inver, CV_8U);
        //cv::imwrite("D:\\inver.png", inver);

        //cv::Mat py_pasted = cv::imread("D:\\2.png");
        cv::add(inver, pasted_face_make_result, upsample_img, cv::noArray(), CV_32F);


        //upsample_img.convertTo(upsample_img, CV_8U);
        //cv::imwrite("D:\\upsample_img_2.png", upsample_img);

    }

//    if (upsample_img.depth() > CV_8U)
      upsample_img.convertTo(upsample_img, CV_8U);

    //if (!save_path.empty())
    //    cv::imwrite(save_path, upsample_img);

    return upsample_img;
}

void CFaceRestoreUtil::back_img_diffiusion(cv::Mat    p_MatImg)
{
    cv::Mat bk_Img = p_MatImg;/*cv::imread(m_strImgPath);*/

    int h = p_MatImg.rows;
    int w = p_MatImg.cols;
    int h_up = h * this->upscale_factor;
    int w_up = w * this->upscale_factor;

    cv::Mat bk_image_upsample(bk_Img);
    cv::resize(bk_Img, bk_image_upsample, cv::Size(w_up, h_up));

    double alpha = 0.01; // Adjust this parameter as per your needs
    double K = 30.0; // Adjust this parameter as per your needs
    int niters = 10; // Adjust this parameter as per your needs

    cv::ximgproc::anisotropicDiffusion(bk_image_upsample, m_diffusedImage, alpha, K, niters);
    //m_diffusedImage = bk_image_upsample;

    //m_diffusedImage.convertTo(m_diffusedImage, CV_8U);
    //cv::imwrite("D:\\diffused_1.png", m_diffusedImage);

    //cv::ximgproc::fastGlobalSmootherFilter(bk_image_upsample, m_diffusedImage, /*parameters*/);

    //m_diffusedImage.convertTo(m_diffusedImage, CV_8U);
    //cv::imwrite("D:\\diffused_2.png", m_diffusedImage);

    return;
}