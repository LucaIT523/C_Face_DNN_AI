
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

//#include <opencv2/opencv.hpp>
//#include <opencv2/face.hpp>
#include "FaceUtil.h"


using namespace std;



// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
std::vector<cv::Point2f>			//. Return - Landmark List.
GetLandmarksOfFace_Dlib(
    cv::Mat			p_Img
,   std::string		p_strModelPath
,   int&            p_nFaceCnt
){

    std::vector<cv::Point2f>        w_Landmarks_List;

    int         w_x = 0;
    int         w_y = 0;
    float       w_fx = 0.0;
    float       w_fy = 0;

    //. 
    p_nFaceCnt = 0;
    //.
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor landmarksDetector;
    p_strModelPath += "//68_face_landmarks.dat";
    dlib::deserialize(p_strModelPath) >> landmarksDetector;
    // Load the input image
    cv::Mat image = p_Img;//cv::imread(p_strImgPath);

    // Convert the image to dlib's format
    dlib::cv_image<dlib::bgr_pixel> dlibImage(image);

    // Detect faces in the image
    std::vector<dlib::rectangle> faces = detector(dlibImage);
    // Iterate over detected faces
    for (const auto& face : faces) {
        // Detect landmarks for each face
        dlib::full_object_detection landmarks = landmarksDetector(dlibImage, face);

        p_nFaceCnt++;

        //. eye-l
        w_x = landmarks.part(36).x() + landmarks.part(39).x();
        w_fx = (float)w_x / 2.0;
        w_y = (landmarks.part(37).y() + landmarks.part(38).y() + landmarks.part(40).y() + landmarks.part(41).y()) ;
        w_fy = (float)w_y / 4.0;
        w_Landmarks_List.push_back(cv::Point2f(w_fx, w_fy));
//        std::cout << w_fx << "  " << w_fy << std::endl;
//        cv::circle(image, cv::Point2f(w_fx, w_fy), 2, cv::Scalar(0, 255, 0), cv::FILLED);

        //. eye-R
        w_x = landmarks.part(42).x() + landmarks.part(45).x();
        w_fx = (float)w_x / 2.0;
        w_y = (landmarks.part(43).y() + landmarks.part(44).y() + landmarks.part(46).y() + landmarks.part(47).y());
        w_fy = (float)w_y / 4.0;
        w_Landmarks_List.push_back(cv::Point2f(w_fx, w_fy));
//        std::cout << w_fx << "  " << w_fy << std::endl;
//        cv::circle(image, cv::Point2f(w_fx, w_fy), 2, cv::Scalar(0, 255, 0), cv::FILLED);

        //. nose
        w_Landmarks_List.push_back(cv::Point2f(landmarks.part(30).x(), landmarks.part(30).y()));
//        cv::circle(image, cv::Point2f(landmarks.part(30).x(), landmarks.part(30).y()), 2, cv::Scalar(0, 255, 0), cv::FILLED);
//        std::cout << landmarks.part(30).x() << "  " << landmarks.part(30).y() << std::endl;


        //. mouth - 2
        w_Landmarks_List.push_back(cv::Point2f(landmarks.part(48).x(), landmarks.part(48).y()));
        w_Landmarks_List.push_back(cv::Point2f(landmarks.part(54).x(), landmarks.part(54).y()));
        //cv::circle(image, cv::Point2f(landmarks.part(48).x(), landmarks.part(48).y()), 2, cv::Scalar(0, 255, 0), cv::FILLED);
        //cv::circle(image, cv::Point2f(landmarks.part(54).x(), landmarks.part(54).y()), 2, cv::Scalar(0, 255, 0), cv::FILLED);
        //std::cout << landmarks.part(48).x() << "  " << landmarks.part(48).y() << std::endl;
        //std::cout << landmarks.part(54).x() << "  " << landmarks.part(54).y() << std::endl;


        //std::stringstream w_strOut;
        //w_strOut << "D:\\land_result" << ".png";
        //cv::imwrite(w_strOut.str().c_str(), image);

        // Draw landmarks on the image
        //for (unsigned int i = 0; i < landmarks.num_parts(); ++i) {
    //        std::cout << landmarks.part(i).x() << "   " << landmarks.part(i).y() << std::endl;
    //        cv::circle(image, cv::Point(landmarks.part(i).x(), landmarks.part(i).y()), 1, cv::Scalar(0, 255, 0), cv::FILLED);
    //        std::stringstream w_strOut;
    //        w_strOut << "D:\\land_" << i << ".png";
    //        cv::imwrite(w_strOut.str().c_str(), image);
        //}
    }

	return w_Landmarks_List;
}

// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
std::vector<cv::Point2f>			//. Return - Landmark List.
GetLandmarksOfFace_Opencv(
    std::string		p_strImgPath
,   int&            p_nFaceCnt
){
    std::vector<cv::Point2f>        w_Landmarks_List;
/*
    // Load the face landmark detection model
    cv::Ptr<cv::face::Facemark> facemark = cv::face::FacemarkLBF::create();
    facemark->loadModel("lbfmodel.yaml");

    // Load the input image
    cv::Mat image = cv::imread("input.jpg");

    // Convert the image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // Perform face detection
    std::vector<cv::Rect> faces;
    cv::CascadeClassifier faceCascade;
    faceCascade.load(cv::samples::findFile("haarcascade_frontalface_default.xml"));
    faceCascade.detectMultiScale(grayImage, faces, 1.1, 3, 0, cv::Size(30, 30));

    // Perform face landmark detection for each detected face
    std::vector<std::vector<cv::Point2f>> landmarks;
    facemark->fit(image, faces, landmarks);

    // Draw the landmarks on the image
    for (const auto& landmark : landmarks)
    {
        for (const auto& point : landmark)
        {
            cv::circle(image, point, 2, cv::Scalar(0, 255, 0), -1);
        }
    }
*/

    return w_Landmarks_List;
}
