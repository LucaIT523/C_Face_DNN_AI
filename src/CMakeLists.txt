cmake_minimum_required(VERSION 3.12)
project(FaceSuperTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(UNIX)
set(CMAKE_GENERATOR "Unix Makefiles")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_BUILD_TYPES "Release" CACHE STRING "" FORCE)
#set(CMAKE_GENERATOR_PLATFORM "x64" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -m64")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# Download and extract libtorch
file(DOWNLOAD "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.1%2Bcpu.zip"
      "${CMAKE_CURRENT_BINARY_DIR}/libtorch.zip"
      SHOW_PROGRESS)

execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_CURRENT_BINARY_DIR}/libtorch.zip"
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

set(LIBTORCH_PATH "${CMAKE_CURRENT_BINARY_DIR}/libtorch")

set(CMAKE_PREFIX_PATH "${LIBTORCH_PATH}")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
find_package(X11 REQUIRED)
find_package(FFmpeg REQUIRED COMPONENTS AVCODEC AVFORMAT AVUTIL SWSCALE)


set(SOURCES
    common1/FaceUtil.cpp
    common1/ImgUtil.cpp
    common1/LoadModel.cpp
    engine/FaceRestoreUtil.cpp
    engine/FaceSuperEng.cpp
    engine/FaceSuperInterface.cpp
    engine/VQAutoEncoder.cpp
    engine/VQAutoEncoderEx.cpp
    engine/FaceParsing.cpp
    common1/ffmpegUtil.cpp
    MyTest.cpp
)

set(HEADERS
    common1/FaceUtil.h
    common1/ImgUtil.h
    common1/LoadModel.h
    common/mydefine.h
    engine/FaceRestoreUtil.hpp
    engine/FaceSuperEng.hpp
    engine/FaceSuperInterface.hpp
    engine/VQAutoEncoder.hpp
    engine/VQAutoEncoderEx.hpp
    common1/ffmpegUtil.h
    engine/FaceParsing.hpp
)

add_executable(FaceSuperTest ${SOURCES} ${HEADERS})

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -D_DEBUG -D_CONSOLE -D_GLIBCXX_USE_CXX11_ABI=0 -Wformat -Wformat-extra-args")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  -DNDEBUG -D_CONSOLE -D_GLIBCXX_USE_CXX11_ABI=0 -Wformat -Wformat-extra-args")

if(WIN32)

target_include_directories(FaceSuperTest PRIVATE
    ./common
    ./common1
    ./engine
    ./model
    ./reflib/opencv47/include
    ./reflib/libtorch-r/include/torch/csrc/api/include
    ./reflib/libtorch-r/include
    ./common1
    ./reflib/dlib
)

target_link_directories(FaceSuperTest PRIVATE
    ./reflib/libtorch-r/lib
    ./reflib/opencv47/lib
    ./reflib/dlib/lib
)

target_link_libraries(FaceSuperTest PRIVATE
    c10.lib
    torch.lib
    torch_cpu.lib
    opencv_core470.lib
    opencv_dnn470.lib
    opencv_features2d470.lib
    opencv_flann470.lib
    opencv_gapi470.lib
    opencv_ml470.lib
    opencv_highgui470.lib
    opencv_imgcodecs470.lib
    opencv_imgproc470.lib
    opencv_videoio470.lib
    opencv_video470.lib
    dlib19.24.99_release_64bit_msvc1933.lib
    opencv_calib3d470.lib
)
elseif(UNIX)

target_include_directories(FaceSuperTest PRIVATE
    ./common
    ./common1
    ./engine
    ./model
    ./common1
    /usr/local/include/opencv4
    ${FFMPEG_INCLUDE_DIR}
)

target_link_directories(FaceSuperTest PRIVATE
    ${LIBTORCH_PATH}/lib
    /usr/local/lib
)

target_link_libraries(FaceSuperTest PRIVATE
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    ${FFMPEG_LIBAVFORMAT_LIBRARY}
    ${FFMPEG_LIBAVCODEC_LIBRARY}
    ${FFMPEG_LIBAVUTIL_LIBRARY}
    ${FFMPEG_LIBSWSCALE_LIBRARY}
    dlib::dlib
    pthread
)
endif()

set_target_properties(FaceSuperTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
)
