
#include "ClaFaceDetectionOpencv.h"
#include "YoloInference.h"


ClaFaceDetectionOpencv::ClaFaceDetectionOpencv()
{

}

ClaFaceDetectionOpencv::~ClaFaceDetectionOpencv()
{

}

//. Face Detection of Yolo8
int		//. Return : 0 - Success
ClaFaceDetectionOpencv::
FaceDectionYolo(wchar_t* p_pszImagePath, std::vector<ST_FaceRectInfo>& p_stRectInfo)
{
	int			w_nRtn = -1;

	Inference	inf("yolov8n.onnx", cv::Size(512, 512));

	// Inference starts here...
	std::vector<Detection> output = inf.runInference(p_pszImagePath);
	cv::Mat faceImg = cv::imread(p_pszImagePath);

	int detections = output.size();

	for (int i = 0; i < detections; ++i)
	{
		Detection detection = output[i];

		cv::Rect box = detection.box;
		cv::Scalar color = detection.color;

		// Detection box
		cv::rectangle(faceImg, box, color, 2);
	}
	// Inference ends here...


	//. OK
	w_nRtn = 0;
L_EXIT:
	return w_nRtn;
}
//. Test Opencv 
int	
ClaFaceDetectionOpencv::
FaceDection(wchar_t* p_pszImagePath, std::vector<ST_FaceRectInfo>& p_stRectInfo)
{
    vector<Rect> faces, faces2;
    Mat gray, smallImg;

    cvtColor(img, gray, COLOR_BGR2GRAY); // Convert to Gray Scale
    double fx = 1 / scale;

    // Resize the Grayscale Image
    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
    equalizeHist(smallImg, smallImg);

    // Detect faces of different sizes using cascade classifier
    cascade.detectMultiScale(smallImg, faces, 1.1,
        2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    // Draw circles around the faces
    for (size_t i = 0; i < faces.size(); i++)
    {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = Scalar(255, 0, 0); // Color for Drawing tool
        int radius;

        double aspect_ratio = (double)r.width / r.height;
        if (0.75 < aspect_ratio && aspect_ratio < 1.3)
        {
            center.x = cvRound((r.x + r.width * 0.5) * scale);
            center.y = cvRound((r.y + r.height * 0.5) * scale);
            radius = cvRound((r.width + r.height) * 0.25 * scale);
            circle(img, center, radius, color, 3, 8, 0);
        }
        else
            rectangle(img, cvPoint(cvRound(r.x * scale), cvRound(r.y * scale)),
                cvPoint(cvRound((r.x + r.width - 1) * scale),
                    cvRound((r.y + r.height - 1) * scale)), color, 3, 8, 0);
        if (nestedCascade.empty())
            continue;
        smallImgROI = smallImg(r);

        // Detection of eyes in the input image
        nestedCascade.detectMultiScale(smallImgROI, nestedObjects, 1.1, 2,
            0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        // Draw circles around eyes
        for (size_t j = 0; j < nestedObjects.size(); j++)
        {
            Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width * 0.5) * scale);
            center.y = cvRound((r.y + nr.y + nr.height * 0.5) * scale);
            radius = cvRound((nr.width + nr.height) * 0.25 * scale);
            circle(img, center, radius, color, 3, 8, 0);
        }
    }

    //. OK
    w_nRtn = 0;
L_EXIT:
    return w_nRtn;

}
