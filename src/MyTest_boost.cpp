// MyTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <memory>
#include "FaceSuperEng.hpp"
#include "FaceSuperInterface.hpp"
#include "FaceUtil.h"
#include "ffmpegUtil.h"
#include "WebserverAPI.hpp"


int main(int argc, char* argv[])
{
 
    net::io_service ioService{ 1 };
    tcp::endpoint endpoint(net::ip::make_address("192.168.1.1"), 12345);
    tcp::acceptor acceptor(ioService, endpoint);

    while (true) {
        tcp::socket socket(ioService);
        acceptor.accept(socket);

        boost::beast::flat_buffer buffer;
        http::request<http::string_body> request;
        boost::beast::http::read(socket, buffer, request);

        http::response<http::file_body> response_file;
        http::response<http::string_body> response_str;

        if (request.method() == http::verb::post && request.target() == "/upload_img") {
            std::string filename = "./uploaded/imgfile"; // Set the desired filename for uploaded image files
            FileUpload(filename, { request.body().begin(), request.body().end() }, response_str);
            beast::http::write(socket, response_str);
        }
        else if (request.method() == http::verb::post && request.target() == "/upload_video") {
            std::string filename = "./uploaded/videofile"; // Set the desired filename for uploaded video files
            FileUpload(filename, { request.body().begin(), request.body().end() }, response_str, 2);
            beast::http::write(socket, response_str);
        }
        else if (request.method() == http::verb::get && request.target() == "/proc_img") {
            std::string filename = request.body();
            Engine_ImageProc(filename, response_str);
            beast::http::write(socket, response_str);
        }
        else if (request.method() == http::verb::get && request.target() == "/proc_video") {
            std::string filename = request.body();
            Engine_VideoProc(filename, response_str);
            beast::http::write(socket, response_str);
        }
        //else if (request.method() == http::verb::get && request.target() == "/download") {
        //    std::string filename = request.body(); //. download file 
        //    FileDownload(filename, response_file);
        //    beast::http::write(socket, response_file);
        //}
        else if (request.method() == http::verb::get && request.target() == "/init") {
            Engine_Init(response_str);
            beast::http::write(socket, response_str);
        }
        else if (request.method() == http::verb::get && request.target() == "/close") {
            Engine_Close(response_str);
            beast::http::write(socket, response_str);
        }
        else {
            response_str.result(http::status::bad_request);
            response_str.set(http::field::content_type, "text/plain");
            response_str.body() = "Invalid request";

            beast::http::write(socket, response_str);
        }

    }

    //ioService.run();

    return GD_SUCCESS;
}
