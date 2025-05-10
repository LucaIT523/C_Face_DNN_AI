#ifndef __WebServer_API_HPP
#define __WebServer_API_HPP

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio.hpp>
#include <iostream>                    // std::cout
#include <fstream>                     // std::ofstream
#include <filesystem>                  // std::filesystem

// Define Namespace
namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>


// Function to handle engine initialization
void Engine_Init(http::response<http::string_body>& response);

// Function to handle image file upload
void FileUpload(std::string& filepath_Header, const std::vector<char>& content, http::response<http::string_body>& response, int	p_nImageFileOpt = 1);

// Function to handle engine start
void Engine_ImageProc(std::string& filename, http::response<http::string_body>& response);

// Function to handle engine start
void Engine_VideoProc(std::string& filename, http::response<http::string_body>& response);

//// Function to handle image file download
//void FileDownload(std::string& filename, http::response<http::file_body>& response);

// Function to handle engine close
void Engine_Close(http::response<http::string_body>& response);



#endif // __WebServer_API_HPP