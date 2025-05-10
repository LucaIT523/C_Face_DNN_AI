
// For Original Header
#include <thread>
#include "WebserverAPI.hpp"
#include "FaceSuperInterface.hpp"
#include "ffmpegUtil.h"

static  int     g_nFileCnt = 0;
HANDLE          g_hEngine = 0x00;
std::string     g_strDownPath = "./download/";
std::string     g_strModelPath = "./model";


void call_proc_thread(std::string  p_filename_IN, std::string  p_filename_OUT, int  p_nImgOpt, int fileCount)
{
    if (p_nImgOpt == 1) {
        FaceSuper_EngProc(g_hEngine, p_filename_IN, p_filename_OUT);
    }
    else {
        std::filesystem::path folderPath = std::filesystem::path(p_filename_IN).parent_path();
        folderPath += std::to_string(fileCount) + "/";
        std::filesystem::create_directory(folderPath);
        std::string audio_temp_file = folderPath.string() + "audio_temp.aac";
        std::string video_temp_file = folderPath.string() + "video_temp.mp4";

        //. 
        Extract_Audio_File(p_filename_IN, audio_temp_file);

        //. Start engine
//        w_nSts = MyFaceSuper_VideoProc(w_Handle, w_strInPath, video_temp_file);
        MyFaceSuper_VideoProc_Ex(g_hEngine, p_filename_IN, video_temp_file);

        std::string runpath = std::filesystem::current_path().string();
#ifdef WIN32
        std::string command = runpath + "\\ffmpeg.exe -i " + video_temp_file + " -i " + audio_temp_file + " -c copy " + p_filename_OUT;
#else
        std::string command = "ffmpeg -i " + video_temp_file + " -i " + audio_temp_file + " -c copy " + p_filename_OUT;
#endif
        system(command.c_str());

        std::remove(audio_temp_file.c_str());
        std::remove(video_temp_file.c_str());
    }
    return;
}


void Engine_Init(http::response<http::string_body>& response) 
{
    //. restoration
    g_hEngine = FaceSuper_Init(1, Set_Device());
    if (g_hEngine != 0) {
    

        FaceSuper_LoadModel(g_hEngine, g_strModelPath);
        //. Create Response code - OK
        response.result(http::status::ok);
        response.set(http::field::content_type, "text/plain");
        //beast::ostream(response.body()) << "OK";
        response.body() = "OK";
    }
    else {
        response.result(http::status::method_not_allowed);
        response.set(http::field::content_type, "text/plain");
        //beast::ostream(response.body()) << "Faild";
        response.body() = "Faild";
    }

    return;
}
// Function to handle engine close
void Engine_Close(http::response<http::string_body>& response)
{

    FaceSuper_Close(g_hEngine);

    //. Create Response code - OK
    response.result(http::status::ok);
    response.set(http::field::content_type, "text/plain");
    //beast::ostream(response.body()) << "OK";
    response.body() = "OK";

}


// Function to handle image file upload
void FileUpload(std::string& filepath_Header, const std::vector<char>& content, http::response<http::string_body>& response, int	p_nImageFileOpt) {
    
    std::string        w_strFilePath_IN = "";
    std::string        w_strFilePath_OUT = "";
    std::string        w_strFileName = "";

    int                w_nCount = g_nFileCnt++;
    //.
    //. image file
    if (p_nImageFileOpt == 1) {
        w_strFilePath_IN = filepath_Header + "_" + std::to_string(g_nFileCnt) + ".png";
        w_strFileName = "imgfile_" + std::to_string(g_nFileCnt) + ".png";
        w_strFilePath_OUT = g_strDownPath + "imgfile_rest_" + std::to_string(g_nFileCnt) + ".png";;
    }
    //. video file
    else if (p_nImageFileOpt == 2) {
        w_strFilePath_IN = filepath_Header + "_" + std::to_string(g_nFileCnt) + ".mp4";
        w_strFileName = "videofile_" + std::to_string(g_nFileCnt) + ".mp4";
        w_strFilePath_OUT = g_strDownPath + "videofile_rest" + std::to_string(g_nFileCnt) + ".mp4";;
    }
    //. unknown file
    else {
        w_strFilePath_IN = "";
    }

    // Handle the image file upload
    std::ofstream output(w_strFilePath_IN, std::ios::binary);
    if (w_strFilePath_IN.length() > 0 && output.is_open() ) {
        output.write(content.data(), content.size());
        output.close();

        //. proc thread
        std::thread(call_proc_thread, w_strFilePath_IN, w_strFilePath_OUT, p_nImageFileOpt, w_nCount);
        //. response 
        response.result(http::status::ok);
        response.set(http::field::content_type, "text/plain");
        response.body() = w_strFileName;
    }
    else {
        response.result(http::status::internal_server_error);
        response.set(http::field::content_type, "text/plain");
        response.body() = "unknown_file";
    }
    return;
}
//
//// Function to handle image file download
//void FileDownload(std::string& filename, http::response<http::file_body>& response)
//{
//    filename = g_strDownPath + filename;
//    std::ifstream input(filename, std::ios::binary);
//    if (input.is_open()) {
//        std::vector<char> content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
//        input.close();
//    
//        response.result(http::status::ok);
//        response.set(http::field::content_type, "application/octet-stream");
//        response.set(http::field::content_length, std::to_string(content.size()));
//        //beast::ostream(response.body()) << content.data();
//        //response.body() = std::move(content);
//        std::cout << "File downloaded successfully!" << std::endl;
//    }
//    else {
//        std::cerr << "Failed to open file for reading." << std::endl;
//        response.result(http::status::not_found);
////        beast::ostream(response.body()) << "unknown_file";
////        response.body() = "unknown_file";
//    }
//
//    return;
//}
// check engine result.
void Engine_ImageProc(std::string& filename, http::response<http::string_body>& response)
{
    std::string     w_strFilePath;
    w_strFilePath = g_strDownPath + filename;

    std::ifstream input(w_strFilePath, std::ios::binary);
    //. if exist , OK
    if (input.is_open()) {
        input.close();

        response.result(http::status::ok);
        response.set(http::field::content_type, "text/plain");
        response.body() = "OK";
    }
    //. 
    else {
        response.result(http::status::processing);
        response.set(http::field::content_type, "text/plain");
        response.body() = "processing";
    }
    return;
}
// check engine result.
void Engine_VideoProc(std::string& filename, http::response<http::string_body>& response)
{
    std::string     w_strFilePath;
    w_strFilePath = g_strDownPath + filename;

    std::ifstream input(w_strFilePath, std::ios::binary);
    //. if exist , OK
    if (input.is_open()) {
        input.close();

        response.result(http::status::ok);
        response.set(http::field::content_type, "text/plain");
        response.body() = "OK";
    }
    //. 
    else {
        response.result(http::status::processing);
        response.set(http::field::content_type, "text/plain");
        response.body() = "processing";
    }
    return;
}




