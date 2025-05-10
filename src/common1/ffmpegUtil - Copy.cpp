
#include "ffmpegUtil.h"



cv::Mat convertFrameToMat(const AVFrame* frame) 
{
    int width = frame->width;
    int height = frame->height;
    cv::Mat image(height, width, CV_8UC3);
    int cvLinesizes[1];
    cvLinesizes[0] = image.step1();
    SwsContext* conversion = sws_getContext(
        width, height, (AVPixelFormat)frame->format, width, height,
        AVPixelFormat::AV_PIX_FMT_BGR24, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    sws_scale(conversion, frame->data, frame->linesize, 0, height, &image.data,  cvLinesizes);
    sws_freeContext(conversion);

    return image;
}

AVFrame* cvmatToAvframe(cv::Mat* image, AVFrame* frame) 
{
#if 0
    int width = image->cols;
    int height = image->rows;
    int cvLinesizes[1];
    cvLinesizes[0] = image->step1();
    if (frame == NULL) {
        frame = av_frame_alloc();
        av_image_alloc(frame->data, frame->linesize, width, height,
            AVPixelFormat::AV_PIX_FMT_YUV420P, 1);
    }
    SwsContext* conversion = sws_getContext(
        width, height, AVPixelFormat::AV_PIX_FMT_BGR24, width, height,
        (AVPixelFormat)frame->format, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    sws_scale(conversion, &image->data, cvLinesizes, 0, height, frame->data,
        frame->linesize);
    sws_freeContext(conversion);

#else 
    int width = image->cols;
    int height = image->rows;
    if (frame == NULL) {
        frame = av_frame_alloc();
        frame->format = AV_PIX_FMT_YUV420P;
        frame->width = width;
        frame->height = height;
        std::cout << "frame->width" << frame->width << "frame->height" << frame->height;

        int res = av_frame_get_buffer(frame, 32);
    }

    cv::cvtColor(*image, *image, cv::COLOR_BGR2YUV_I420);
    std::memcpy(frame->data[0], image->data, width * height);
    std::memcpy(frame->data[1], image->data + width * height, width * height / 4);
    std::memcpy(frame->data[2], image->data + width * height * 5 / 4, width * height / 4);


#endif
    return frame;
}
// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------
int
Extract_Audio_File(
	std::string		p_strVideoFile_IN
,	std::string		p_strAudioFile_OUT
){
    // Open the input video file
    AVFormatContext* formatContext = nullptr;
    if (avformat_open_input(&formatContext, p_strVideoFile_IN.c_str(), nullptr, nullptr) != 0) {
        std::cerr << "Failed to open input video file" << std::endl;
        return -1;
    }

    // Retrieve stream information
    if (avformat_find_stream_info(formatContext, nullptr) < 0) {
        std::cerr << "Failed to retrieve stream information" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    // Find the audio stream
    int audioStreamIndex = -1;
    for (unsigned int i = 0; i < formatContext->nb_streams; ++i) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audioStreamIndex = i;
            break;
        }
    }

    if (audioStreamIndex == -1) {
        std::cerr << "Failed to find audio stream" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    // Get a pointer to the audio codec context
    AVCodecContext* codecContext = avcodec_alloc_context3(nullptr);
    if (!codecContext) {
        std::cerr << "Failed to allocate audio codec context" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    if (avcodec_parameters_to_context(codecContext, formatContext->streams[audioStreamIndex]->codecpar) < 0) {
        std::cerr << "Failed to copy codec parameters to audio codec context" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return -1;
    }

    // Find the audio decoder
    const AVCodec* codec = avcodec_find_decoder(codecContext->codec_id);
    if (!codec) {
        std::cerr << "Failed to find audio decoder" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return -1;
    }

    // Open the audio codec
    if (avcodec_open2(codecContext, codec, nullptr) < 0) {
        std::cerr << "Failed to open audio codec" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return -1;
    }

    // Create an output file for the audio
    AVPacket packet;
    //av_init_packet(&packet);
    packet.data = nullptr;
    packet.size = 0;

    AVFormatContext* outputFormatContext = nullptr;
    if (avformat_alloc_output_context2(&outputFormatContext, nullptr, nullptr, p_strAudioFile_OUT.c_str()) < 0) {
        std::cerr << "Failed to create output audio file" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return -1;
    }

    // Add a new audio stream to the output file
    AVStream* outputAudioStream = avformat_new_stream(outputFormatContext, codec);
    if (!outputAudioStream) {
        std::cerr << "Failed to create output audio stream" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        avformat_free_context(outputFormatContext);
        return -1;
    }

    // Copy codec parameters from the input audio stream to the output audio stream
    if (avcodec_parameters_copy(outputAudioStream->codecpar, formatContext->streams[audioStreamIndex]->codecpar) < 0) {
        std::cerr << "Failed to copy codec parameters from input to output audio stream" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        avformat_free_context(outputFormatContext);
        return -1;
    }

    // Open the output audio file for writing
    if (!(outputFormatContext->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&outputFormatContext->pb, p_strAudioFile_OUT.c_str(), AVIO_FLAG_WRITE) < 0) {
            std::cerr << "Failed to open output audio file for writing" << std::endl;
            avcodec_free_context(&codecContext);
            avformat_close_input(&formatContext);
            avformat_free_context(outputFormatContext);
            return -1;
        }
    }

    // Write the output audio file header
    if (avformat_write_header(outputFormatContext, nullptr) < 0) {
        std::cerr << "Failed to write output audio file header" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        avformat_free_context(outputFormatContext);
        return -1;
    }

    // Read frames from the input video file and write them to the output audio file
    while (av_read_frame(formatContext, &packet) >= 0) {
        if (packet.stream_index == audioStreamIndex) {
            // Write the audio packet to the output audio file
            packet.stream_index = outputAudioStream->index;
            av_interleaved_write_frame(outputFormatContext, &packet);
        }

        av_packet_unref(&packet);
    }

    // Write the output audio file trailer
    av_write_trailer(outputFormatContext);

    // Free resources
    avcodec_free_context(&codecContext);
    avformat_close_input(&formatContext);
    avformat_free_context(outputFormatContext);

    return 0;
}

int  FaceSuper_VideoProc(HANDLE p_h, std::string	p_strInImg, std::string	p_strOutImg)
{
    int             w_nSts = GD_UNKNOWN_ERR;
    char* w_sbuff = (char*)p_h;
    int             w_nDetFaceCnt = 0;

    //. Check Handle
    if (p_h == 0) {
        return GD_INIT_ERR;
    }

    // Open the input video file
    AVFormatContext* formatContext = nullptr;
    if (avformat_open_input(&formatContext, p_strInImg.c_str(), nullptr, nullptr) != 0) {
        std::cerr << "Failed to open input video file" << std::endl;
        return -1;
    }

    // Retrieve stream information
    if (avformat_find_stream_info(formatContext, nullptr) < 0) {
        std::cerr << "Failed to retrieve stream information" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    // Find the video stream
    int videoStreamIndex = -1;
    for (unsigned int i = 0; i < formatContext->nb_streams; ++i) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            break;
        }
    }

    if (videoStreamIndex == -1) {
        std::cerr << "Failed to find video stream" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    // Get the video codec parameters
    AVCodecParameters* videoCodecParams = formatContext->streams[videoStreamIndex]->codecpar;

    // Find the video decoder
    const AVCodec* codec = avcodec_find_decoder(videoCodecParams->codec_id);
    if (!codec) {
        std::cerr << "Failed to find video decoder" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    // Allocate a codec context
    AVCodecContext* codecContext = avcodec_alloc_context3(codec);
    if (!codecContext) {
        std::cerr << "Failed to allocate video codec context" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    // Copy codec parameters to codec context
    if (avcodec_parameters_to_context(codecContext, videoCodecParams) < 0) {
        std::cerr << "Failed to copy codec parameters to video codec context" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return -1;
    }

    // Open the video codec
    if (avcodec_open2(codecContext, codec, nullptr) < 0) {
        std::cerr << "Failed to open video codec" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return -1;
    }

    // Create a new video file for the demuxed video
    AVFormatContext* outputFormatContext = nullptr;
    if (avformat_alloc_output_context2(&outputFormatContext, nullptr, nullptr, p_strOutImg.c_str()) < 0) {
        std::cerr << "Failed to create output video file" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return -1;
    }

    // Add a new video stream to the output file
    AVStream* outputVideoStream = avformat_new_stream(outputFormatContext, codec);
    if (!outputVideoStream) {
        std::cerr << "Failed to create output video stream" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        avformat_free_context(outputFormatContext);
        return -1;
    }

    // Copy codec parameters from input to output video stream
    if (avcodec_parameters_copy(outputVideoStream->codecpar, videoCodecParams) < 0) {
        std::cerr << "Failed to copy codec parameters from input to output video stream" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        avformat_free_context(outputFormatContext);
        return -1;
    }

    // Change the frame size in the output video stream
    outputVideoStream->codecpar->width *= 2;
    outputVideoStream->codecpar->height *= 2;

    // Open the output video file for writing
    if (!(outputFormatContext->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&outputFormatContext->pb, "output_video.mp4", AVIO_FLAG_WRITE) < 0) {
            std::cerr << "Failed to open output video file for writing" << std::endl;
            avcodec_free_context(&codecContext);
            avformat_close_input(&formatContext);
            avformat_free_context(outputFormatContext);
            return -1;
        }
    }

    // Write the output video file header
    if (avformat_write_header(outputFormatContext, nullptr) < 0) {
        std::cerr << "Failed to write output video file header" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        avformat_free_context(outputFormatContext);
        return -1;
    }

    // Read packets from the input video file, demux video packets, change frame size, and write them to the output video file
    AVPacket packet;
    packet.size = 0;
    while (av_read_frame(formatContext, &packet) >= 0) {
        if (packet.stream_index == videoStreamIndex) {
            // Change the frame size
            AVPacket outputPacket = packet;
            AVFrame* frame = av_frame_alloc();
            int ret = avcodec_send_packet(codecContext, &packet);
            if (ret == 0) {
                ret = avcodec_receive_frame(codecContext, frame);
                //. error ???
//                ret = avcodec_send_frame(codecContext, frame);
                if (ret == 0) {
                    
                    cv::Mat     w_ImgMat_IN = convertFrameToMat(frame);
                    cv::Mat     w_ImgMat_Out;
                    OneFrameOfVideoProc(w_ImgMat_IN, w_ImgMat_Out);

                    cv::imwrite("D:\\aaaa.png", w_ImgMat_Out);
                    cvmatToAvframe(&w_ImgMat_Out, frame);
                    /*
                    frame->width *= 2;
                    frame->height *= 2;

                    // Allocate frame buffer for the changed frame size
                    int bufferSize = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, frame->width, frame->height, 1);
                    uint8_t* buffer = (uint8_t*)av_malloc(bufferSize);
                    av_image_fill_arrays(frame->data, frame->linesize, buffer, AV_PIX_FMT_YUV420P, frame->width, frame->height, 1);

                    // Encode and write the changed video frame to the output video file
//                    av_init_packet(&outputPacket);
                    outputPacket.size = 0;

                    ret = avcodec_send_frame(codecContext, frame);
                    av_strerror(ret, error_str, 256);
                    if (ret == 0) {
                        ret = avcodec_receive_packet(codecContext, &outputPacket);
                        if (ret == 0) {
                            outputPacket.stream_index = outputVideoStream->index;
                            av_interleaved_write_frame(outputFormatContext, &outputPacket);
                        }
                    }

                    av_packet_unref(&outputPacket);
                    av_free(buffer);
                    */
                }
            }

            av_frame_free(&frame);
        }
        //. Errror ????
        //av_packet_unref(&packet);
        packet.size = 0;
    }

    // Write the output video file trailer
    av_write_trailer(outputFormatContext);

    // Free resources
    avformat_close_input(&formatContext);
    avformat_free_context(outputFormatContext);
    avcodec_free_context(&codecContext);

    //. 
    w_nSts = 0;
    return w_nSts;
}


static void encode(AVCodecContext *enc_ctx, AVFrame *frame, AVPacket *pkt,
                   FILE *outfile)
{
    int ret;

    /* send the frame to the encoder */
    if (frame)
        printf("Send frame %d \n", frame->pts);

    ret = avcodec_send_frame(enc_ctx, frame);
    if (ret < 0) {
        fprintf(stderr, "Error sending a frame for encoding\n");
        exit(1);
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            fprintf(stderr, "Error during encoding\n");
            exit(1);
        }

        printf("Write packet %3\"PRId64\" (size=%5d)\n", pkt->pts, pkt->size);
        fwrite(pkt->data, 1, pkt->size, outfile);
        av_packet_unref(pkt);
    }
}

int
Enhance_Video_File(std::string	p_strVideoFile_IN, std::string	p_strVideoFile_OUT)
{
    char        error_str[256] = "";
    const char* pInFileName = p_strVideoFile_IN.c_str();
    const char* pOutFileName = p_strVideoFile_OUT.c_str();

    FILE *pOutFileHandle;
    uint8_t endcode[] = { 0, 0, 1, 0xb7 };
    const char  pOutCodecName[] = "libx264"; //libx264
    AVCodecContext *pOutCodecCtx= NULL;
    const AVCodec* pOutCodec = nullptr;
    // Open the input video file
    AVFormatContext* formatContext = nullptr;
    AVPacket* pOutputPacket;

    if (avformat_open_input(&formatContext, pInFileName, nullptr, nullptr) != 0) {
        std::cerr << "Failed to open input video file" << std::endl;
        return -1;
    }

    // Retrieve stream information
    if (avformat_find_stream_info(formatContext, nullptr) < 0) {
        std::cerr << "Failed to retrieve stream information" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    // Find the video stream
    int videoStreamIndex = -1;
    for (unsigned int i = 0; i < formatContext->nb_streams; ++i) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            break;
        }
    }

    if (videoStreamIndex == -1) {
        std::cerr << "Failed to find video stream" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    // Get the video codec parameters
    AVCodecParameters* videoCodecParams = formatContext->streams[videoStreamIndex]->codecpar;

    // Find the video decoder
    const AVCodec* codec = avcodec_find_decoder(videoCodecParams->codec_id);
    if (!codec) {
        std::cerr << "Failed to find video decoder" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    // Allocate a codec context
    AVCodecContext* pInCodecCtx = avcodec_alloc_context3(codec);
    if (!pInCodecCtx) {
        std::cerr << "Failed to allocate video codec context" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    // Copy codec parameters to codec context
    if (avcodec_parameters_to_context(pInCodecCtx, videoCodecParams) < 0) {
        std::cerr << "Failed to copy codec parameters to video codec context" << std::endl;
        avcodec_free_context(&pInCodecCtx);
        avformat_close_input(&formatContext);
        return -1;
    }

    // Open the video codec
    if (avcodec_open2(pInCodecCtx, codec, nullptr) < 0) {
        std::cerr << "Failed to open video codec" << std::endl;
        avcodec_free_context(&pInCodecCtx);
        avformat_close_input(&formatContext);
        return -1;
    }



     /* find the mpeg1video encoder */
    pOutCodec = avcodec_find_encoder_by_name(pOutCodecName);
    if (!pOutCodec) {
        fprintf(stderr, "Codec '%s' not found\n", pOutCodecName);
        exit(1);
    }
    pOutCodecCtx = avcodec_alloc_context3(pOutCodec);

   // pOutCodecCtx = avcodec_alloc_context3(pOutCodec);
    if (!pOutCodecCtx) {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }

    /* put sample parameters */
    pOutCodecCtx->bit_rate = 400000;
    /* resolution must be a multiple of two */
    pOutCodecCtx->width = pInCodecCtx->width;
    pOutCodecCtx->height = pInCodecCtx->height;
    /* frames per second */
    pOutCodecCtx->time_base.num = 1;
    pOutCodecCtx->time_base.den = 25;

  //  pOutCodecCtx->framerate.num = 25;
//pOutCodecCtx->framerate.den = 1;

   // pOutCodecCtx->framerate = (AVRational){25, 1};

    /* emit one intra frame every ten frames
     * check frame pict_type before passing frame
     * to encoder, if frame->pict_type is AV_PICTURE_TYPE_I
     * then gop_size is ignored and the output of encoder
     * will always be I frame irrespective to gop_size
     */
    pOutCodecCtx->gop_size = 10;
    pOutCodecCtx->max_b_frames = 0;
    pOutCodecCtx->pix_fmt = AV_PIX_FMT_YUV420P;

    if (pOutCodec->id == AV_CODEC_ID_H264)
        av_opt_set(pOutCodecCtx->priv_data, "preset", "slow", 0);

    /* open it */
    if (avcodec_open2(pOutCodecCtx, pOutCodec, NULL) < 0) {
        fprintf(stderr, "Could not open codec: %s\n");
        exit(1);
    }

    pOutFileHandle = fopen(pOutFileName, "wb");
    if (!pOutFileHandle) {
        fprintf(stderr, "Could not open %s\n", pOutFileName);
        exit(1);
    }


    // Read packets from the input video file, demux video packets, change frame size, and write them to the output video file
    pOutputPacket = av_packet_alloc();
    AVPacket inpacket;
    inpacket.size = 0;
    while (av_read_frame(formatContext, &inpacket) >= 0) {
        if (inpacket.stream_index == videoStreamIndex) {
            // Change the frame size
            AVFrame* frame = av_frame_alloc();
            int ret = avcodec_send_packet(pInCodecCtx, &inpacket);
            if (ret == 0) {
                ret = avcodec_receive_frame(pInCodecCtx, frame);
                if (ret == 0) {
                    //convertFrameToMat(frame);

                    // Allocate frame buffer for the changed frame size
                    //int bufferSize = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, frame->width, frame->height, 1);
                    //uint8_t* buffer = (uint8_t*)av_malloc(bufferSize);
                    //(frame->data, frame->linesize, buffer, AV_PIX_FMT_YUV420P, frame->width, frame->height, 1);

                    // Encode and write the changed video frame to the output video file
//                    av_init_packet(&outputPacket);
                    pOutputPacket->size = 0;
                    encode(pOutCodecCtx, frame, pOutputPacket, pOutFileHandle);
                }
            }

            av_frame_free(&frame);
        }
        //. Errror ????
        //av_packet_unref(&packet);
        inpacket.size = 0;
    }

    encode(pOutCodecCtx, nullptr, pOutputPacket, pOutFileHandle);

    if (codec->id == AV_CODEC_ID_MPEG1VIDEO || codec->id == AV_CODEC_ID_MPEG2VIDEO)
        fwrite(endcode, 1, sizeof(endcode), pOutFileHandle);
    fclose(pOutFileHandle);

    // Free resources
    avformat_close_input(&formatContext);
    avcodec_free_context(&pInCodecCtx);
    avcodec_free_context(&pOutCodecCtx);
    av_packet_free(&pOutputPacket);

    return 0;
}

int  
MyFaceSuper_VideoProc(HANDLE p_h, std::string	p_strInImg, std::string	p_strOutImg)
{

     char        error_str[256] = "";
    const char* pInFileName = p_strInImg.c_str();
    const char* pOutFileName = p_strOutImg.c_str();


    int             w_nSts = GD_UNKNOWN_ERR;
    char* w_sbuff = (char*)p_h;
    int             w_nDetFaceCnt = 0;

    //. Check Handle
    if (p_h == 0) {
        return GD_INIT_ERR;
    }


    FILE *pOutFileHandle;
    uint8_t endcode[] = { 0, 0, 1, 0xb7 };
    const char  pOutCodecName[] = "libx264"; //libx264
    AVCodecContext *pOutCodecCtx= NULL;
    const AVCodec* pOutCodec = nullptr;
    // Open the input video file
    AVFormatContext* formatContext = nullptr;
    AVPacket* pOutputPacket;

    if (avformat_open_input(&formatContext, pInFileName, nullptr, nullptr) != 0) {
        std::cerr << "Failed to open input video file" << std::endl;
        return -1;
    }

    // Retrieve stream information
    if (avformat_find_stream_info(formatContext, nullptr) < 0) {
        std::cerr << "Failed to retrieve stream information" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    // Find the video stream
    int videoStreamIndex = -1;
    for (unsigned int i = 0; i < formatContext->nb_streams; ++i) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            break;
        }
    }

    if (videoStreamIndex == -1) {
        std::cerr << "Failed to find video stream" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    // Get the video codec parameters
    AVCodecParameters* videoCodecParams = formatContext->streams[videoStreamIndex]->codecpar;

    // Find the video decoder
    const AVCodec* codec = avcodec_find_decoder(videoCodecParams->codec_id);
    if (!codec) {
        std::cerr << "Failed to find video decoder" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    // Allocate a codec context
    AVCodecContext* pInCodecCtx = avcodec_alloc_context3(codec);
    if (!pInCodecCtx) {
        std::cerr << "Failed to allocate video codec context" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    // Copy codec parameters to codec context
    if (avcodec_parameters_to_context(pInCodecCtx, videoCodecParams) < 0) {
        std::cerr << "Failed to copy codec parameters to video codec context" << std::endl;
        avcodec_free_context(&pInCodecCtx);
        avformat_close_input(&formatContext);
        return -1;
    }

    // Open the video codec
    if (avcodec_open2(pInCodecCtx, codec, nullptr) < 0) {
        std::cerr << "Failed to open video codec" << std::endl;
        avcodec_free_context(&pInCodecCtx);
        avformat_close_input(&formatContext);
        return -1;
    }



     /* find the mpeg1video encoder */
    pOutCodec = avcodec_find_encoder_by_name(pOutCodecName);
    if (!pOutCodec) {
        fprintf(stderr, "Codec '%s' not found\n", pOutCodecName);
        exit(1);
    }
    pOutCodecCtx = avcodec_alloc_context3(pOutCodec);

   // pOutCodecCtx = avcodec_alloc_context3(pOutCodec);
    if (!pOutCodecCtx) {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }

    /* put sample parameters */
    pOutCodecCtx->bit_rate = 400000;
    /* resolution must be a multiple of two */
    pOutCodecCtx->width = pInCodecCtx->width*2;
    pOutCodecCtx->height = pInCodecCtx->height*2;
    /* frames per second */
    pOutCodecCtx->time_base.num = 1;
    pOutCodecCtx->time_base.den = 25;

  //  pOutCodecCtx->framerate.num = 25;
//pOutCodecCtx->framerate.den = 1;

    pOutCodecCtx->gop_size = 10;
    pOutCodecCtx->max_b_frames = 1;
    pOutCodecCtx->pix_fmt = AV_PIX_FMT_YUV420P;

    if (pOutCodec->id == AV_CODEC_ID_H264)
        av_opt_set(pOutCodecCtx->priv_data, "preset", "slow", 0);

    /* open it */
    if (avcodec_open2(pOutCodecCtx, pOutCodec, NULL) < 0) {
        fprintf(stderr, "Could not open codec: %s\n");
        exit(1);
    }

    pOutFileHandle = fopen(pOutFileName, "wb");
    if (!pOutFileHandle) {
        fprintf(stderr, "Could not open %s\n", pOutFileName);
        exit(1);
    }


    // Read packets from the input video file, demux video packets, change frame size, and write them to the output video file
    pOutputPacket = av_packet_alloc();
    AVPacket inpacket;
    inpacket.size = 0;
    while (av_read_frame(formatContext, &inpacket) >= 0) {
        if (inpacket.stream_index == videoStreamIndex) {
            // Change the frame size
            AVFrame* frame = av_frame_alloc();
            AVFrame* pEncFrame = nullptr;

            int ret = avcodec_send_packet(pInCodecCtx, &inpacket);
            if (ret == 0) {
                ret = avcodec_receive_frame(pInCodecCtx, frame);
                if (ret == 0) {
                    //convertFrameToMat(frame);
                    cv::Mat     w_ImgMat_IN = convertFrameToMat(frame);
                    cv::Mat     w_ImgMat_Out;
                    OneFrameOfVideoProc(w_ImgMat_IN, w_ImgMat_Out);

                   // cv::imwrite("D:\\aaaa.png", w_ImgMat_Out);
                   pEncFrame = cvmatToAvframe(&w_ImgMat_Out, pEncFrame);
                    
                    //return 0;
                    // Allocate frame buffer for the changed frame size
                   //  fprintf(stderr, "Encoding process %d\n");
                   // pEncFrame = av_frame_alloc();

                   // pEncFrame->width = frame->width * 2;
                   // pEncFrame->height = frame->height * 2;
                   // pEncFrame->format = AV_PIX_FMT_YUV420P;
                   // int res = av_frame_get_buffer(pEncFrame, 32);

                    //int bufferSize = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, pEncFrame->width, pEncFrame->height, 1);
                    // uint8_t* buffer = (uint8_t*)av_malloc(bufferSize);
                   // av_image_fill_arrays(pEncFrame->data, pEncFrame->linesize, buffer, AV_PIX_FMT_YUV420P, pEncFrame->width, pEncFrame->height, 1);


                    pEncFrame->pts = frame->pts;
                    pOutputPacket->size = 0;
                    encode(pOutCodecCtx, pEncFrame, pOutputPacket, pOutFileHandle);

                    static int ii = 0;
                    fprintf(stderr, "Encoding process %d\n", ++ii);

                }
            }

            av_frame_free(&frame);
            if (pEncFrame) {
                av_frame_free(&pEncFrame);
            }

        }
        //. Errror ????
        //av_packet_unref(&packet);
        inpacket.size = 0;
    }

    encode(pOutCodecCtx, nullptr, pOutputPacket, pOutFileHandle);

    if (codec->id == AV_CODEC_ID_MPEG1VIDEO || codec->id == AV_CODEC_ID_MPEG2VIDEO)
        fwrite(endcode, 1, sizeof(endcode), pOutFileHandle);
    fclose(pOutFileHandle);

    // Free resources
    avformat_close_input(&formatContext);
    avcodec_free_context(&pInCodecCtx);
    avcodec_free_context(&pOutCodecCtx);
    av_packet_free(&pOutputPacket);

    return 0;
}

int
MyEnhance_Video_File(std::string	p_strVideoFile_IN, std::string	p_strVideoFile_OUT)
{
    const char *filename, *codec_name;
    const AVCodec *codec;
    AVCodecContext *c= NULL;
    int i, ret, x, y;
    FILE *f;
    AVFrame *frame;
    AVPacket *pkt;
    uint8_t endcode[] = { 0, 0, 1, 0xb7 };

    filename = p_strVideoFile_OUT.c_str();
    codec_name = "MPEG2VIDEO";

    /* find the mpeg1video encoder */
    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        fprintf(stderr, "Codec '%s' not found\n", codec_name);
        exit(1);
    }

    c = avcodec_alloc_context3(codec);
    if (!c) {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }

    pkt = av_packet_alloc();
    if (!pkt)
        exit(1);

    /* put sample parameters */
    c->bit_rate = 400000;
    /* resolution must be a multiple of two */
    c->width = 352;
    c->height = 288;
    /* frames per second */
    c->time_base.num = 1;
    c->time_base.den = 25;

        c->framerate.num = 25;
    c->framerate.den = 1;

    /* emit one intra frame every ten frames
     * check frame pict_type before passing frame
     * to encoder, if frame->pict_type is AV_PICTURE_TYPE_I
     * then gop_size is ignored and the output of encoder
     * will always be I frame irrespective to gop_size
     */
    c->gop_size = 10;
    c->max_b_frames = 1;
    c->pix_fmt = AV_PIX_FMT_YUV420P;

    if (codec->id == AV_CODEC_ID_H264)
        av_opt_set(c->priv_data, "preset", "slow", 0);

    /* open it */
    ret = avcodec_open2(c, codec, NULL);
    if (ret < 0) {
        fprintf(stderr, "Could not open codec: %d\n" , ret);
        exit(1);
    }

    f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", filename);
        exit(1);
    }

    frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate video frame\n");
        exit(1);
    }
    frame->format = c->pix_fmt;
    frame->width  = c->width;
    frame->height = c->height;

    ret = av_frame_get_buffer(frame, 0);
    if (ret < 0) {
        fprintf(stderr, "Could not allocate the video frame data\n");
        exit(1);
    }

    /* encode 1 second of video */
    for (i = 0; i < 50; i++) {
        fflush(stdout);

        /* Make sure the frame data is writable.
           On the first round, the frame is fresh from av_frame_get_buffer()
           and therefore we know it is writable.
           But on the next rounds, encode() will have called
           avcodec_send_frame(), and the codec may have kept a reference to
           the frame in its internal structures, that makes the frame
           unwritable.
           av_frame_make_writable() checks that and allocates a new buffer
           for the frame only if necessary.
         */
        ret = av_frame_make_writable(frame);
        if (ret < 0)
            exit(1);

        /* Prepare a dummy image.
           In real code, this is where you would have your own logic for
           filling the frame. FFmpeg does not care what you put in the
           frame.
         */
        /* Y */
        for (y = 0; y < c->height; y++) {
            for (x = 0; x < c->width; x++) {
                frame->data[0][y * frame->linesize[0] + x] = x + y + i * 3;
            }
        }

        /* Cb and Cr */
        for (y = 0; y < c->height/2; y++) {
            for (x = 0; x < c->width/2; x++) {
                frame->data[1][y * frame->linesize[1] + x] = 128 + y + i * 2;
                frame->data[2][y * frame->linesize[2] + x] = 64 + x + i * 5;
            }
        }

        frame->pts = i;

        /* encode the image */
        encode(c, frame, pkt, f);
    }

    /* flush the encoder */
    encode(c, NULL, pkt, f);

    /* Add sequence end code to have a real MPEG file.
       It makes only sense because this tiny examples writes packets
       directly. This is called "elementary stream" and only works for some
       codecs. To create a valid file, you usually need to write packets
       into a proper file format or protocol; see mux.c.
     */
    if (codec->id == AV_CODEC_ID_MPEG1VIDEO || codec->id == AV_CODEC_ID_MPEG2VIDEO)
        fwrite(endcode, 1, sizeof(endcode), f);
    fclose(f);

    avcodec_free_context(&c);
    av_frame_free(&frame);
    av_packet_free(&pkt);

    return 0;
}
