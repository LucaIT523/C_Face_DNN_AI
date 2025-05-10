

#include <iostream>
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <mediapipe/framework/graph.h>
#include <mediapipe/framework/packet.h>
#include <mediapipe/framework/port/opencv_highgui_inc.h>
#include <mediapipe/framework/port/opencv_imgproc_inc.h>
#include "ClaFaceDetectionMediapipe.h"


ClaFaceDetectionMediapipe::ClaFaceDetectionMediapipe()
{

}

ClaFaceDetectionMediapipe::~ClaFaceDetectionMediapipe()
{

}

//. 
int	
ClaFaceDetectionMediapipe::
FaceDection(wchar_t* p_pszImagePath, std::vector<ST_FaceRectInfo>& p_stRectInfo)
{
    int            w_nRtn = 0;

    // Initialize the calculator graph.
    mediapipe::CalculatorGraphConfig graph_config;
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(graph_config));

    // Load the face detection module.
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    // Open a video capture stream.
    cv::Mat frame;

    // Wrap the OpenCV image in an ImageFrame.
    mediapipe::ImageFrame image_frame =  mediapipe::ImageFrame::FromMat(frame, mediapipe::ImageFormat::SRGB);

    // Create an input packet and send it to the graph.
    mediapipe::Packet packet =  mediapipe::Adopt(image_frame.release()).At(mediapipe::Timestamp(0));
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream("input_frame", packet));
/*
    // Get the output packets from the graph.
    mediapipe::Packet output_packet;
    if (!graph.GetPacketFromOutputStream("output_detections", &output_packet)
        .ok()) {
        break;
    }

    // Convert the packet to a vector of face detection results.
    std::vector<mediapipe::NormalizedRect> detections =
        output_packet.Get<std::vector<mediapipe::NormalizedRect>>();

    // Draw bounding boxes around the detected faces.
    for (const auto& detection : detections) {
        cv::Rect rect(detection.x_center() * frame.cols,
            detection.y_center() * frame.rows,
            detection.width() * frame.cols,
            detection.height() * frame.rows);
        cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
    }
*/
    // Shut down the graph.
//    MP_RETURN_IF_ERROR(graph.CloseInputStream("input_frame"));
//    return graph.WaitUntilDone();

    //. OK
    w_nRtn = 0;
L_EXIT:
    return w_nRtn;

}
