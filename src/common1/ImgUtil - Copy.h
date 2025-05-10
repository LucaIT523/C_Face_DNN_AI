#ifndef IMG_UTIL_InferenceH
#define IMG_UTIL_InferenceH
//---------------------------------------------------------------------------

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "torch/script.h"
#include "torch/torch.h"

using namespace cv;
using namespace std;
namespace nn = torch::nn;
/*
    Args:
        imgs(list[ndarray] | ndarray) : Input images.
        bgr2rgb(bool) : Whether to change bgr to rgb.
        float32(bool) : Whether to change to float32.

    Returns :
        list[tensor] | tensor : Tensor images.If returned results only have
        one element, just return tensor.
*/
torch::Tensor img2tensor(cv::Mat img, bool bgr2rgb, bool float32);


/*
    Convert torch Tensors into image numpy arrays.

    After clamping to[min, max], values will be normalized to[0, 1].

    Args:
        tensor(Tensor or list[Tensor]) : Accept shapes :
        1) 4D mini - batch Tensor of shape(B x 3 / 1 x H x W);
        2) 3D Tensor of shape(3 / 1 x H x W);
        3) 2D Tensor of shape(H x W).
        Tensor channel should be in RGB order.
        rgb2bgr(bool) : Whether to change rgb to bgr.
        out_type(numpy type) : output types.If ``np.uint8``, transform outputs
        to uint8 type with range[0, 255]; otherwise, float type with
        range[0, 1].Default: ``np.uint8``.
        min_max(tuple[int]) : min and max values for clamp.

    Returns :
        (Tensor or list) : 3D ndarray of shape(H x W x C) OR 2D ndarray of
        shape(H x W).The channel order is BGR.
*/
cv::Mat tensor2img(torch::Tensor tensor, bool rgb2bgr = true, int out_type = CV_8U, std::pair<float, float> min_max = { 0, 1 });


//. custom normalize function of torchvision
void normalize_custom(torch::Tensor p_Tensor);

#endif // IMG_UTIL_InferenceH