# Find FFmpeg libraries
# This module sets the following variables:
#    FFMPEG_FOUND       - True if FFmpeg is found
#    FFMPEG_INCLUDE_DIR - The directory containing FFmpeg headers
#    FFMPEG_LIBRARIES   - The FFmpeg libraries to link against

find_path(FFMPEG_INCLUDE_DIR libavformat/avformat.h)

find_library(FFMPEG_LIBAVFORMAT_LIBRARY avformat)
find_library(FFMPEG_LIBAVCODEC_LIBRARY avcodec)
find_library(FFMPEG_LIBAVUTIL_LIBRARY avutil)
find_library(FFMPEG_LIBSWSCALE_LIBRARY swscale)

if(FFMPEG_INCLUDE_DIR AND FFMPEG_LIBAVFORMAT_LIBRARY AND FFMPEG_LIBAVCODEC_LIBRARY AND FFMPEG_LIBAVUTIL_LIBRARY AND FFMPEG_LIBSWSCALE_LIBRARY)
    set(FFMPEG_FOUND TRUE)
else()
    set(FFMPEG_FOUND FALSE)
endif()

if(FFMPEG_FOUND)
    message(STATUS "Found FFmpeg: ${FFMPEG_LIBAVFORMAT_LIBRARY}")
else()
    message(WARNING "FFmpeg not found")
endif()

mark_as_advanced(FFMPEG_LIBAVFORMAT_LIBRARY FFMPEG_LIBAVCODEC_LIBRARY FFMPEG_LIBAVUTIL_LIBRARY FFMPEG_LIBSWSCALE_LIBRARY)