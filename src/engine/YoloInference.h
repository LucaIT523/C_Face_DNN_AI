#ifndef InferenceH
#define InferenceH
//---------------------------------------------------------------------------

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

//. Information of face detection
struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};
//. Face detection using Yolov8 (Yolov8.onnx face model)
class YoloInference
{
public:
    YoloInference(const std::string &onnxModelPath, const cv::Size &modelInputShape = {640, 640}, const std::string &classesTxtFile = "", const bool &runWithCuda = false);
    std::vector<Detection> runInference(const cv::Mat &input);

private:
    void loadClassesFromFile();
    void loadOnnxNetwork();
    cv::Mat formatToSquare(const cv::Mat &source);

    std::string modelPath{};
    std::string classesPath{};
    bool cudaEnabled{};

    cv::Size2f modelShape{};

    float modelConfidenseThreshold {0.25};
    float modelScoreThreshold      {0.45};
    float modelNMSThreshold        {0.50};

	bool letterBoxForSquare = true;

    cv::dnn::Net net;
};

#endif // INFERENCE_H
