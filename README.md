<div align="center">
   <h1>C++_Face_DNN_AI</h1>
</div>

-----------------------------------------------------
The code is a C++  project for a neural network module built using the LibTorch (C++ PyTorch) library. It defines classes for deep learning layers and models used for facial image enhancement tasks such as restoration, colorization, and inpainting.



Folder Structure (Linux and Windows for C++)
-----------------------------------------------------

bin	->		 results (exe, dll)

src	-> 		 source

common	->	 common function (define, log, struct info, ...)

common1	->	 Face Utility,  FFmpeg Utility,  Image Utility, Load Model Function

engine	-> 	Face Detection, Parsing, Restoration and neural network using torch


model	->	pretrained model, restoration , colorization,  inpainting,  face parsing, face detection landmarks(5, 68)	

reflib	->	dlib, Opencv, Libtorch, yolo







-----------------------------------------------------
Face Detection Engine 
-----------------------------------------------------
1. Dlib 
2. opencv

-----------------------------------------------------
Face Parsing Engine 
-----------------------------------------------------
1. Resnet50



-----------------------------------------------------
Face Super DNN Engine 
-----------------------------------------------------
restoration

inpainting

colorization



-----------------------------------------------------
Background Unsampler Engine 
-----------------------------------------------------
OpenCV



Main Function 
-----------------------------------------------------

<div align="center">
   <img src=https://github.com/LucaIT523/C_Face_DNN_AI/blob/main/src/images/1.png>
</div> 		




Face Colorization
-----------------------------------------------------

<div align="center">
   <img src=https://github.com/LucaIT523/C_Face_DNN_AI/blob/main/src/images/2.png>
</div> 		





Face Restoration 
-----------------------------------------------------

<div align="center">
   <img src=https://github.com/LucaIT523/C_Face_DNN_AI/blob/main/src/images/3.png>
</div> 		



