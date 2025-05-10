


#include <stdlib.h>
#include<stdio.h>
#include<cstdlib>
#include<iostream>
#include<string.h>
#include<fstream>

#include "ffmpegUtil.h"


namespace fs = std::filesystem;

//
//#define STREAM_DURATION   10.0
//#define STREAM_FRAME_RATE 25 /* 25 images/s */
//#define STREAM_PIX_FMT    AV_PIX_FMT_YUV420P /* default pix_fmt */
//
//#define SCALE_FLAGS SWS_BICUBIC
//
//// a wrapper around a single output AVStream
//typedef struct OutputStream {
//    AVStream* st;
//    AVCodecContext* enc;
//
//    /* pts of the next frame that will be generated */
//    int64_t next_pts;
//    int samples_count;
//
//    AVFrame* frame;
//    AVFrame* tmp_frame;
//
//    AVPacket* tmp_pkt;
//
//    float t, tincr, tincr2;
//
//    struct SwsContext* sws_ctx;
//    struct SwrContext* swr_ctx;
//} OutputStream;
//
////static void log_packet(const AVFormatContext* fmt_ctx, const AVPacket* pkt)
////{
////    AVRational* time_base = &fmt_ctx->streams[pkt->stream_index]->time_base;
////
////    printf("pts:%s pts_time:%s dts:%s dts_time:%s duration:%s duration_time:%s stream_index:%d\n",
////        av_ts2str(pkt->pts), av_ts2timestr(pkt->pts, time_base),
////        av_ts2str(pkt->dts), av_ts2timestr(pkt->dts, time_base),
////        av_ts2str(pkt->duration), av_ts2timestr(pkt->duration, time_base),
////        pkt->stream_index);
////}
//
//static int write_frame(AVFormatContext* fmt_ctx, AVCodecContext* c,
//    AVStream* st, AVFrame* frame, AVPacket* pkt)
//{
//    int ret;
//
//    // send the frame to the encoder
//    ret = avcodec_send_frame(c, frame);
//    if (ret < 0) {
//        fprintf(stderr, "Error sending a frame to the encoder: \n");
//        return ret;
//    }
//
//    while (ret >= 0) {
//        ret = avcodec_receive_packet(c, pkt);
//        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
//            break;
//        else if (ret < 0) {
//            fprintf(stderr, "Error encoding a frame: \n");
//            return ret;
//        }
//
//        /* rescale output packet timestamp values from codec to stream timebase */
//        av_packet_rescale_ts(pkt, c->time_base, st->time_base);
//        pkt->stream_index = st->index;
//
//        /* Write the compressed frame to the media file. */
////        log_packet(fmt_ctx, pkt);
//        ret = av_interleaved_write_frame(fmt_ctx, pkt);
//        /* pkt is now blank (av_interleaved_write_frame() takes ownership of
//         * its contents and resets pkt), so that no unreferencing is necessary.
//         * This would be different if one used av_write_frame(). */
//        if (ret < 0) {
//            fprintf(stderr, "Error while writing output packet: \n");
//            return ret;
//        }
//    }
//
//    return ret == AVERROR_EOF ? 1 : 0;
//}
//
///* Add an output stream. */
//static void add_stream(OutputStream* ost, AVFormatContext* oc,
//    const AVCodec** codec,
//    enum AVCodecID codec_id)
//{
//    AVCodecContext* c;
//    int i;
//    AVRational  w_AVRational;
//    AVChannelLayout  src;
//    src.order = AV_CHANNEL_ORDER_NATIVE;
//    src.nb_channels = (2);
//    src.u.mask =  (AV_CH_LAYOUT_STEREO);
//
//    /* find the encoder */
//    *codec = avcodec_find_encoder(codec_id);
//    if (!(*codec)) {
//        fprintf(stderr, "Could not find encoder for '%s'\n",
//            avcodec_get_name(codec_id));
//        return;
//    }
//
//    ost->tmp_pkt = av_packet_alloc();
//    if (!ost->tmp_pkt) {
//        fprintf(stderr, "Could not allocate AVPacket\n");
//        return;
//    }
//
//    ost->st = avformat_new_stream(oc, NULL);
//    if (!ost->st) {
//        fprintf(stderr, "Could not allocate stream\n");
//        return;
//    }
//    ost->st->id = oc->nb_streams - 1;
//    c = avcodec_alloc_context3(*codec);
//    if (!c) {
//        fprintf(stderr, "Could not alloc an encoding context\n");
//        return;
//    }
//    ost->enc = c;
//
//    switch ((*codec)->type) {
//    case AVMEDIA_TYPE_AUDIO:
//        c->sample_fmt = (*codec)->sample_fmts ?
//            (*codec)->sample_fmts[0] : AV_SAMPLE_FMT_FLTP;
//        c->bit_rate = 64000;
//        c->sample_rate = 44100;
//        if ((*codec)->supported_samplerates) {
//            c->sample_rate = (*codec)->supported_samplerates[0];
//            for (i = 0; (*codec)->supported_samplerates[i]; i++) {
//                if ((*codec)->supported_samplerates[i] == 44100)
//                    c->sample_rate = 44100;
//            }
//        }
//        w_AVRational.num = 1;
//        w_AVRational.den = c->sample_rate;
//        av_channel_layout_copy(&c->ch_layout, &src);
//        ost->st->time_base = w_AVRational;
//        break;
//
//    case AVMEDIA_TYPE_VIDEO:
//        c->codec_id = codec_id;
//
//        c->bit_rate = 400000;
//        /* Resolution must be a multiple of two. */
//        c->width = 352;
//        c->height = 288;
//        /* timebase: This is the fundamental unit of time (in seconds) in terms
//         * of which frame timestamps are represented. For fixed-fps content,
//         * timebase should be 1/framerate and timestamp increments should be
//         * identical to 1. */
//        w_AVRational.num = 1;
//        w_AVRational.den = STREAM_FRAME_RATE;
//        ost->st->time_base = w_AVRational;
//        c->time_base = ost->st->time_base;
//
//        c->gop_size = 12; /* emit one intra frame every twelve frames at most */
//        c->pix_fmt = STREAM_PIX_FMT;
//        if (c->codec_id == AV_CODEC_ID_MPEG2VIDEO) {
//            /* just for testing, we also add B-frames */
//            c->max_b_frames = 2;
//        }
//        if (c->codec_id == AV_CODEC_ID_MPEG1VIDEO) {
//            /* Needed to avoid using macroblocks in which some coeffs overflow.
//             * This does not happen with normal video, it just happens here as
//             * the motion of the chroma plane does not match the luma plane. */
//            c->mb_decision = 2;
//        }
//        break;
//
//    default:
//        break;
//    }
//
//    /* Some formats want stream headers to be separate. */
//    if (oc->oformat->flags & AVFMT_GLOBALHEADER)
//        c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
//}
//
///**************************************************************/
///* audio output */
//
//static AVFrame* alloc_audio_frame(enum AVSampleFormat sample_fmt,
//    const AVChannelLayout* channel_layout,
//    int sample_rate, int nb_samples)
//{
//    AVFrame* frame = av_frame_alloc();
//    if (!frame) {
//        fprintf(stderr, "Error allocating an audio frame\n");
//        return nullptr;
//    }
//
//    frame->format = sample_fmt;
//    av_channel_layout_copy(&frame->ch_layout, channel_layout);
//    frame->sample_rate = sample_rate;
//    frame->nb_samples = nb_samples;
//
//    if (nb_samples) {
//        if (av_frame_get_buffer(frame, 0) < 0) {
//            fprintf(stderr, "Error allocating an audio buffer\n");
//            return nullptr;
//        }
//    }
//
//    return frame;
//}
//
//static void open_audio(AVFormatContext* oc, const AVCodec* codec,
//    OutputStream* ost, AVDictionary* opt_arg)
//{
//    AVCodecContext* c;
//    int nb_samples;
//    int ret;
//    AVDictionary* opt = NULL;
//
//    c = ost->enc;
//
//    /* open it */
//    av_dict_copy(&opt, opt_arg, 0);
//    ret = avcodec_open2(c, codec, &opt);
//    av_dict_free(&opt);
//    if (ret < 0) {
//        fprintf(stderr, "Could not open audio codec: \n");
//        return;
//    }
//
//    /* init signal generator */
//    ost->t = 0;
//    ost->tincr = 2 * M_PI * 110.0 / c->sample_rate;
//    /* increment frequency by 110 Hz per second */
//    ost->tincr2 = 2 * M_PI * 110.0 / c->sample_rate / c->sample_rate;
//
//    if (c->codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE)
//        nb_samples = 10000;
//    else
//        nb_samples = c->frame_size;
//
//    ost->frame = alloc_audio_frame(c->sample_fmt, &c->ch_layout,
//        c->sample_rate, nb_samples);
//    ost->tmp_frame = alloc_audio_frame(AV_SAMPLE_FMT_S16, &c->ch_layout,
//        c->sample_rate, nb_samples);
//
//    /* copy the stream parameters to the muxer */
//    ret = avcodec_parameters_from_context(ost->st->codecpar, c);
//    if (ret < 0) {
//        fprintf(stderr, "Could not copy the stream parameters\n");
//        return;
//    }
//
//    /* create resampler context */
//    ost->swr_ctx = swr_alloc();
//    if (!ost->swr_ctx) {
//        fprintf(stderr, "Could not allocate resampler context\n");
//        return;
//    }
//
//    /* set options */
//    av_opt_set_chlayout(ost->swr_ctx, "in_chlayout", &c->ch_layout, 0);
//    av_opt_set_int(ost->swr_ctx, "in_sample_rate", c->sample_rate, 0);
//    av_opt_set_sample_fmt(ost->swr_ctx, "in_sample_fmt", AV_SAMPLE_FMT_S16, 0);
//    av_opt_set_chlayout(ost->swr_ctx, "out_chlayout", &c->ch_layout, 0);
//    av_opt_set_int(ost->swr_ctx, "out_sample_rate", c->sample_rate, 0);
//    av_opt_set_sample_fmt(ost->swr_ctx, "out_sample_fmt", c->sample_fmt, 0);
//
//    /* initialize the resampling context */
//    if ((ret = swr_init(ost->swr_ctx)) < 0) {
//        fprintf(stderr, "Failed to initialize the resampling context\n");
//        return;
//    }
//
//    return;
//}
//
///* Prepare a 16 bit dummy audio frame of 'frame_size' samples and
// * 'nb_channels' channels. */
//static AVFrame* get_audio_frame(OutputStream* ost)
//{
//    AVFrame* frame = ost->tmp_frame;
//    int j, i, v;
//    int16_t* q = (int16_t*)frame->data[0];
//    AVRational  w_AVRational;
//    w_AVRational.num = 1;
//    w_AVRational.den = 1;
//
//    /* check if we want to generate more frames */
//    if (av_compare_ts(ost->next_pts, ost->enc->time_base,
//        STREAM_DURATION, w_AVRational) > 0)
//        return NULL;
//
//    for (j = 0; j < frame->nb_samples; j++) {
//        v = (int)(sin(ost->t) * 10000);
//        for (i = 0; i < ost->enc->ch_layout.nb_channels; i++)
//            *q++ = v;
//        ost->t += ost->tincr;
//        ost->tincr += ost->tincr2;
//    }
//
//    frame->pts = ost->next_pts;
//    ost->next_pts += frame->nb_samples;
//
//    return frame;
//}
//
///*
// * encode one audio frame and send it to the muxer
// * return 1 when encoding is finished, 0 otherwise
// */
//static int write_audio_frame(AVFormatContext* oc, OutputStream* ost)
//{
//    AVCodecContext* c;
//    AVFrame* frame;
//    int ret;
//    int dst_nb_samples;
//    AVRational  w_AVRational;
//
//    c = ost->enc;
//
//    frame = get_audio_frame(ost);
//
//    if (frame) {
//        /* convert samples from native format to destination codec format, using the resampler */
//        /* compute destination number of samples */
//        dst_nb_samples = av_rescale_rnd(swr_get_delay(ost->swr_ctx, c->sample_rate) + frame->nb_samples,
//            c->sample_rate, c->sample_rate, AV_ROUND_UP);
//        av_assert0(dst_nb_samples == frame->nb_samples);
//
//        /* when we pass a frame to the encoder, it may keep a reference to it
//         * internally;
//         * make sure we do not overwrite it here
//         */
//        ret = av_frame_make_writable(ost->frame);
//        if (ret < 0)
//            return ret;
//
//        /* convert to destination format */
//        ret = swr_convert(ost->swr_ctx,
//            ost->frame->data, dst_nb_samples,
//            (const uint8_t**)frame->data, frame->nb_samples);
//        if (ret < 0) {
//            fprintf(stderr, "Error while converting\n");
//            return ret;
//        }
//        frame = ost->frame;
//
//        w_AVRational.num = 1;
//        w_AVRational.den = c->sample_rate;
//        frame->pts = av_rescale_q(ost->samples_count, w_AVRational, c->time_base);
//        ost->samples_count += dst_nb_samples;
//    }
//
//    return write_frame(oc, c, ost->st, frame, ost->tmp_pkt);
//}
//
///**************************************************************/
///* video output */
//
//static AVFrame* alloc_picture(enum AVPixelFormat pix_fmt, int width, int height)
//{
//    AVFrame* picture;
//    int ret;
//
//    picture = av_frame_alloc();
//    if (!picture)
//        return NULL;
//
//    picture->format = pix_fmt;
//    picture->width = width;
//    picture->height = height;
//
//    /* allocate the buffers for the frame data */
//    ret = av_frame_get_buffer(picture, 0);
//    if (ret < 0) {
//        fprintf(stderr, "Could not allocate frame data.\n");
//        return NULL;
//    }
//
//    return picture;
//}
//
//static void open_video(AVFormatContext* oc, const AVCodec* codec,
//    OutputStream* ost, AVDictionary* opt_arg)
//{
//    int ret;
//    AVCodecContext* c = ost->enc;
//    AVDictionary* opt = NULL;
//
//    av_dict_copy(&opt, opt_arg, 0);
//
//    /* open the codec */
//    ret = avcodec_open2(c, codec, &opt);
//    av_dict_free(&opt);
//    if (ret < 0) {
//        fprintf(stderr, "Could not open video codec: \n");
//        return;
//    }
//
//    /* allocate and init a re-usable frame */
//    ost->frame = alloc_picture(c->pix_fmt, c->width, c->height);
//    if (!ost->frame) {
//        fprintf(stderr, "Could not allocate video frame\n");
//        return;
//    }
//
//    /* If the output format is not YUV420P, then a temporary YUV420P
//     * picture is needed too. It is then converted to the required
//     * output format. */
//    ost->tmp_frame = NULL;
//    if (c->pix_fmt != AV_PIX_FMT_YUV420P) {
//        ost->tmp_frame = alloc_picture(AV_PIX_FMT_YUV420P, c->width, c->height);
//        if (!ost->tmp_frame) {
//            fprintf(stderr, "Could not allocate temporary picture\n");
//            return;
//        }
//    }
//
//    /* copy the stream parameters to the muxer */
//    ret = avcodec_parameters_from_context(ost->st->codecpar, c);
//    if (ret < 0) {
//        fprintf(stderr, "Could not copy the stream parameters\n");
//        return;
//    }
//    return;
//}
//
///* Prepare a dummy image. */
//static void fill_yuv_image(AVFrame* pict, int frame_index,
//    int width, int height)
//{
//    int x, y, i;
//
//    i = frame_index;
//
//    /* Y */
//    for (y = 0; y < height; y++)
//        for (x = 0; x < width; x++)
//            pict->data[0][y * pict->linesize[0] + x] = x + y + i * 3;
//
//    /* Cb and Cr */
//    for (y = 0; y < height / 2; y++) {
//        for (x = 0; x < width / 2; x++) {
//            pict->data[1][y * pict->linesize[1] + x] = 128 + y + i * 2;
//            pict->data[2][y * pict->linesize[2] + x] = 64 + x + i * 5;
//        }
//    }
//}
//
//static AVFrame* get_video_frame(OutputStream* ost)
//{
//    AVCodecContext* c = ost->enc;
//    AVRational      w_AVRational;
//    w_AVRational.num = 1;
//    w_AVRational.den = 1;
//    /* check if we want to generate more frames */
//    if (av_compare_ts(ost->next_pts, c->time_base,
//        STREAM_DURATION, w_AVRational) > 0)
//        return NULL;
//
//    /* when we pass a frame to the encoder, it may keep a reference to it
//     * internally; make sure we do not overwrite it here */
//    if (av_frame_make_writable(ost->frame) < 0)
//        return NULL;
//
//    if (c->pix_fmt != AV_PIX_FMT_YUV420P) {
//        /* as we only generate a YUV420P picture, we must convert it
//         * to the codec pixel format if needed */
//        if (!ost->sws_ctx) {
//            ost->sws_ctx = sws_getContext(c->width, c->height,
//                AV_PIX_FMT_YUV420P,
//                c->width, c->height,
//                c->pix_fmt,
//                SCALE_FLAGS, NULL, NULL, NULL);
//            if (!ost->sws_ctx) {
//                fprintf(stderr,
//                    "Could not initialize the conversion context\n");
//                return NULL;
//            }
//        }
//        fill_yuv_image(ost->tmp_frame, ost->next_pts, c->width, c->height);
//        sws_scale(ost->sws_ctx, (const uint8_t* const*)ost->tmp_frame->data,
//            ost->tmp_frame->linesize, 0, c->height, ost->frame->data,
//            ost->frame->linesize);
//    }
//    else {
//        fill_yuv_image(ost->frame, ost->next_pts, c->width, c->height);
//    }
//
//    ost->frame->pts = ost->next_pts++;
//
//    return ost->frame;
//}
//
///*
// * encode one video frame and send it to the muxer
// * return 1 when encoding is finished, 0 otherwise
// */
//static int write_video_frame(AVFormatContext* oc, OutputStream* ost)
//{
//    return write_frame(oc, ost->enc, ost->st, get_video_frame(ost), ost->tmp_pkt);
//}
//
//static void close_stream(AVFormatContext* oc, OutputStream* ost)
//{
//    avcodec_free_context(&ost->enc);
//    av_frame_free(&ost->frame);
//    av_frame_free(&ost->tmp_frame);
//    av_packet_free(&ost->tmp_pkt);
//    sws_freeContext(ost->sws_ctx);
//    swr_free(&ost->swr_ctx);
//}

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
Muxer_Video_Audio(std::string	p_strVideoFile_IN, std::string	p_strAudioFile_IN, std::string p_strMuxerFile)
{
 /*   AVFormatContext* videoFormatContext = nullptr;
    AVFormatContext* audioFormatContext = nullptr;
    AVFormatContext* outputFormatContext = nullptr;
    AVPacket packet;

    if (avformat_open_input(&videoFormatContext, p_strVideoFile_IN.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Failed to open video file.\n";
        return 1;
    }

    if (avformat_find_stream_info(videoFormatContext, nullptr) < 0) {
        std::cerr << "Failed to retrieve input video stream information.\n";
        return 1;
    }

    if (avformat_open_input(&audioFormatContext, p_strAudioFile_IN.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Failed to open audio file.\n";
        return 1;
    }

    if (avformat_find_stream_info(audioFormatContext, nullptr) < 0) {
        std::cerr << "Failed to retrieve input audio stream information.\n";
        return 1;
    }

    if (avformat_alloc_output_context2(&outputFormatContext, nullptr, nullptr, p_strMuxerFile.c_str()) < 0) {
        std::cerr << "Failed to allocate output format context.\n";
        return 1;
    }

    for (unsigned int i = 0; i < videoFormatContext->nb_streams; i++) {
        AVStream* inStream = videoFormatContext->streams[i];
        AVStream* outStream = avformat_new_stream(outputFormatContext, inStream->codec);
        if (!outStream) {
            std::cerr << "Failed to allocate output video stream.\n";
            return 1;
        }

        if (avcodec_copy_context(outStream->codec, inStream->codec) < 0) {
            std::cerr << "Failed to copy video codec context.\n";
            return 1;
        }

        outStream->codec->codec_tag = 0;
        if (outputFormatContext->oformat->flags & AVFMT_GLOBALHEADER)
            outStream->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    for (unsigned int i = 0; i < audioFormatContext->nb_streams; i++) {
        AVStream* inStream = audioFormatContext->streams[i];
        AVStream* outStream = avformat_new_stream(outputFormatContext, inStream->codec->codec);
        if (!outStream) {
            std::cerr << "Failed to allocate output audio stream.\n";
            return 1;
        }

        if (avcodec_copy_context(outStream->codec, inStream->codec) < 0) {
            std::cerr << "Failed to copy audio codec context.\n";
            return 1;
        }

        outStream->codec->codec_tag = 0;
        if (outputFormatContext->oformat->flags & AVFMT_GLOBALHEADER)
            outStream->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    if (!(outputFormatContext->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&outputFormatContext->pb, outputFile, AVIO_FLAG_WRITE) < 0) {
            std::cerr << "Failed to open output file.\n";
            return 1;
        }
    }

    if (avformat_write_header(outputFormatContext, nullptr) < 0) {
        std::cerr << "Failed to write output file header.\n";
        return 1;
    }

    while (true) {
        if (av_read_frame(videoFormatContext, &packet) < 0)
            break;

        packet.stream_index = 0;
        av_interleaved_write_frame(outputFormatContext, &packet);
        av_packet_unref(&packet);
    }

    while (true) {
        if (av_read_frame(audioFormatContext, &packet) < 0)
            break;

        packet.stream_index = 1;
        av_interleaved_write_frame(outputFormatContext, &packet);
        av_packet_unref(&packet);
    }

    av_write_trailer(outputFormatContext);

    avformat_close_input(&videoFormatContext);
    avformat_close_input(&audioFormatContext);

    if (outputFormatContext && !(outputFormatContext->oformat->flags & AVFMT_NOFILE))
        avio_closep(&outputFormatContext->pb);

    avformat_free_context(outputFormatContext);*/


    AVFormatContext* outputFormatContext = nullptr;
    if (avformat_alloc_output_context2(&outputFormatContext, nullptr, nullptr, p_strMuxerFile.c_str()) < 0) {
        std::cerr << "Failed to allocate output format context.\n";
        return 1;
    }

    AVOutputFormat* outputFormat = (AVOutputFormat*)(outputFormatContext->oformat);

    // Open video file
    AVFormatContext* videoFormatContext = nullptr;
    if (avformat_open_input(&videoFormatContext, p_strVideoFile_IN.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Failed to open video file.\n";
        return 1;
    }

    if (avformat_find_stream_info(videoFormatContext, nullptr) < 0) {
        std::cerr << "Failed to retrieve input video stream information.\n";
        return 1;
    }

    // Open audio file
    AVFormatContext* audioFormatContext = nullptr;
    if (avformat_open_input(&audioFormatContext, p_strAudioFile_IN.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Failed to open audio file.\n";
        return 1;
    }

    if (avformat_find_stream_info(audioFormatContext, nullptr) < 0) {
        std::cerr << "Failed to retrieve input audio stream information.\n";
        return 1;
    }

    // Add video stream to output context
    for (unsigned int i = 0; i < videoFormatContext->nb_streams; i++) {
        AVStream* stream = avformat_new_stream(outputFormatContext, nullptr);
        if (!stream) {
            std::cerr << "Failed to allocate output video stream.\n";
            return 1;
        }

        if (avcodec_parameters_copy(stream->codecpar, videoFormatContext->streams[i]->codecpar) < 0) {
            std::cerr << "Failed to copy video codec parameters.\n";
            return 1;
        }
    }

    // Add audio stream to output context
    for (unsigned int i = 0; i < audioFormatContext->nb_streams; i++) {
        AVStream* stream = avformat_new_stream(outputFormatContext, nullptr);
        if (!stream) {
            std::cerr << "Failed to allocate output audio stream.\n";
            return 1;
        }

        if (avcodec_parameters_copy(stream->codecpar, audioFormatContext->streams[i]->codecpar) < 0) {
            std::cerr << "Failed to copy audio codec parameters.\n";
            return 1;
        }
    }

    // Open output file
    if (!(outputFormat->flags & AVFMT_NOFILE)) {
        if (avio_open(&outputFormatContext->pb, p_strMuxerFile.c_str(), AVIO_FLAG_WRITE) < 0) {
            std::cerr << "Failed to open output file.\n";
            return 1;
        }
    }

    // Write header to output file
    if (avformat_write_header(outputFormatContext, nullptr) < 0) {
        std::cerr << "Failed to write output file header.\n";
        return 1;
    }

    // Muxing loop
    AVPacket packet;
    while (true) {
        AVStream* inStream = nullptr;
        AVStream* outStream = nullptr;

        // Read packet from video file
        if (av_read_frame(videoFormatContext, &packet) < 0) {
            break;
        }

        inStream = videoFormatContext->streams[packet.stream_index];
        outStream = outputFormatContext->streams[packet.stream_index];

        // Set correct timestamps for video packet
        packet.pts = av_rescale_q_rnd(packet.pts, inStream->time_base, outStream->time_base, AV_ROUND_NEAR_INF);
        packet.dts = av_rescale_q_rnd(packet.dts, inStream->time_base, outStream->time_base, AV_ROUND_NEAR_INF);
        packet.duration = av_rescale_q(packet.duration, inStream->time_base, outStream->time_base);
        packet.pos = -1;

        // Write video packet to output file
        if (av_interleaved_write_frame(outputFormatContext, &packet) < 0) {
            std::cerr << "Failed to write video packet.\n";
            return 1;
        }

        av_packet_unref(&packet);
    }

    // Muxing loop for audio
    while (true) {
        AVStream* inStream = nullptr;
        AVStream* outStream = nullptr;

        // Read packet from audio file
        if (av_read_frame(audioFormatContext, &packet) < 0) {
            break;
        }

        inStream = audioFormatContext->streams[packet.stream_index];
        outStream = outputFormatContext->streams[packet.stream_index];

        // Set correct timestamps for audio packet
        packet.pts = av_rescale_q_rnd(packet.pts, inStream->time_base, outStream->time_base, AV_ROUND_NEAR_INF);
        packet.dts = av_rescale_q_rnd(packet.dts, inStream->time_base, outStream->time_base, AV_ROUND_NEAR_INF);
        packet.duration = av_rescale_q(packet.duration, inStream->time_base, outStream->time_base);
        packet.pos = -1;

        // Write audio packet to output file
        if (av_interleaved_write_frame(outputFormatContext, &packet) < 0) {
            std::cerr << "Failed to write audio packet.\n";
            return 1;
        }

        av_packet_unref(&packet);
    }

    // Write trailer to output file
    if (av_write_trailer(outputFormatContext) < 0) {
        std::cerr << "Failed to write output file trailer.\n";
        return 1;
    }

    // Close video file
    avformat_close_input(&videoFormatContext);

    // Close audio file
    avformat_close_input(&audioFormatContext);

    // Close output file
    if (outputFormatContext && !(outputFormat->flags & AVFMT_NOFILE)) {
        avio_closep(&outputFormatContext->pb);
    }

    // Free output format context
    avformat_free_context(outputFormatContext);

    return 0;
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
//
//int  FaceSuper_VideoProc(HANDLE p_h, std::string	p_strInImg, std::string	p_strOutImg)
//{
//    int             w_nSts = GD_UNKNOWN_ERR;
//    char* w_sbuff = (char*)p_h;
//    int             w_nDetFaceCnt = 0;
//
//    //. Check Handle
//    if (p_h == 0) {
//        return GD_INIT_ERR;
//    }
//
//    // Open the input video file
//    AVFormatContext* formatContext = nullptr;
//    if (avformat_open_input(&formatContext, p_strInImg.c_str(), nullptr, nullptr) != 0) {
//        std::cerr << "Failed to open input video file" << std::endl;
//        return -1;
//    }
//
//    // Retrieve stream information
//    if (avformat_find_stream_info(formatContext, nullptr) < 0) {
//        std::cerr << "Failed to retrieve stream information" << std::endl;
//        avformat_close_input(&formatContext);
//        return -1;
//    }
//
//    // Find the video stream
//    int videoStreamIndex = -1;
//    for (unsigned int i = 0; i < formatContext->nb_streams; ++i) {
//        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
//            videoStreamIndex = i;
//            break;
//        }
//    }
//
//    if (videoStreamIndex == -1) {
//        std::cerr << "Failed to find video stream" << std::endl;
//        avformat_close_input(&formatContext);
//        return -1;
//    }
//
//    // Get the video codec parameters
//    AVCodecParameters* videoCodecParams = formatContext->streams[videoStreamIndex]->codecpar;
//
//    // Find the video decoder
//    const AVCodec* codec = avcodec_find_decoder(videoCodecParams->codec_id);
//    if (!codec) {
//        std::cerr << "Failed to find video decoder" << std::endl;
//        avformat_close_input(&formatContext);
//        return -1;
//    }
//
//    // Allocate a codec context
//    AVCodecContext* codecContext = avcodec_alloc_context3(codec);
//    if (!codecContext) {
//        std::cerr << "Failed to allocate video codec context" << std::endl;
//        avformat_close_input(&formatContext);
//        return -1;
//    }
//
//    // Copy codec parameters to codec context
//    if (avcodec_parameters_to_context(codecContext, videoCodecParams) < 0) {
//        std::cerr << "Failed to copy codec parameters to video codec context" << std::endl;
//        avcodec_free_context(&codecContext);
//        avformat_close_input(&formatContext);
//        return -1;
//    }
//
//    // Open the video codec
//    if (avcodec_open2(codecContext, codec, nullptr) < 0) {
//        std::cerr << "Failed to open video codec" << std::endl;
//        avcodec_free_context(&codecContext);
//        avformat_close_input(&formatContext);
//        return -1;
//    }
//
//    // Create a new video file for the demuxed video
//    AVFormatContext* outputFormatContext = nullptr;
//    if (avformat_alloc_output_context2(&outputFormatContext, nullptr, nullptr, p_strOutImg.c_str()) < 0) {
//        std::cerr << "Failed to create output video file" << std::endl;
//        avcodec_free_context(&codecContext);
//        avformat_close_input(&formatContext);
//        return -1;
//    }
//
//    // Add a new video stream to the output file
//    AVStream* outputVideoStream = avformat_new_stream(outputFormatContext, codec);
//    if (!outputVideoStream) {
//        std::cerr << "Failed to create output video stream" << std::endl;
//        avcodec_free_context(&codecContext);
//        avformat_close_input(&formatContext);
//        avformat_free_context(outputFormatContext);
//        return -1;
//    }
//
//    // Copy codec parameters from input to output video stream
//    if (avcodec_parameters_copy(outputVideoStream->codecpar, videoCodecParams) < 0) {
//        std::cerr << "Failed to copy codec parameters from input to output video stream" << std::endl;
//        avcodec_free_context(&codecContext);
//        avformat_close_input(&formatContext);
//        avformat_free_context(outputFormatContext);
//        return -1;
//    }
//
//    // Change the frame size in the output video stream
//    outputVideoStream->codecpar->width *= 2;
//    outputVideoStream->codecpar->height *= 2;
//
//    // Open the output video file for writing
//    if (!(outputFormatContext->oformat->flags & AVFMT_NOFILE)) {
//        if (avio_open(&outputFormatContext->pb, "output_video.mp4", AVIO_FLAG_WRITE) < 0) {
//            std::cerr << "Failed to open output video file for writing" << std::endl;
//            avcodec_free_context(&codecContext);
//            avformat_close_input(&formatContext);
//            avformat_free_context(outputFormatContext);
//            return -1;
//        }
//    }
//
//    // Write the output video file header
//    if (avformat_write_header(outputFormatContext, nullptr) < 0) {
//        std::cerr << "Failed to write output video file header" << std::endl;
//        avcodec_free_context(&codecContext);
//        avformat_close_input(&formatContext);
//        avformat_free_context(outputFormatContext);
//        return -1;
//    }
//
//    // Read packets from the input video file, demux video packets, change frame size, and write them to the output video file
//    AVPacket packet;
//    packet.size = 0;
//    while (av_read_frame(formatContext, &packet) >= 0) {
//        if (packet.stream_index == videoStreamIndex) {
//            // Change the frame size
//            AVPacket outputPacket = packet;
//            AVFrame* frame = av_frame_alloc();
//            int ret = avcodec_send_packet(codecContext, &packet);
//            if (ret == 0) {
//                ret = avcodec_receive_frame(codecContext, frame);
//                //. error ???
////                ret = avcodec_send_frame(codecContext, frame);
//                if (ret == 0) {
//                    
//                    cv::Mat     w_ImgMat_IN = convertFrameToMat(frame);
//                    cv::Mat     w_ImgMat_Out;
//                    OneFrameOfVideoProc(w_ImgMat_IN, w_ImgMat_Out);
//
//                    cv::imwrite("D:\\aaaa.png", w_ImgMat_Out);
//                    cvmatToAvframe(&w_ImgMat_Out, frame);
//                    /*
//                    frame->width *= 2;
//                    frame->height *= 2;
//
//                    // Allocate frame buffer for the changed frame size
//                    int bufferSize = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, frame->width, frame->height, 1);
//                    uint8_t* buffer = (uint8_t*)av_malloc(bufferSize);
//                    av_image_fill_arrays(frame->data, frame->linesize, buffer, AV_PIX_FMT_YUV420P, frame->width, frame->height, 1);
//
//                    // Encode and write the changed video frame to the output video file
////                    av_init_packet(&outputPacket);
//                    outputPacket.size = 0;
//
//                    ret = avcodec_send_frame(codecContext, frame);
//                    av_strerror(ret, error_str, 256);
//                    if (ret == 0) {
//                        ret = avcodec_receive_packet(codecContext, &outputPacket);
//                        if (ret == 0) {
//                            outputPacket.stream_index = outputVideoStream->index;
//                            av_interleaved_write_frame(outputFormatContext, &outputPacket);
//                        }
//                    }
//
//                    av_packet_unref(&outputPacket);
//                    av_free(buffer);
//                    */
//                }
//            }
//
//            av_frame_free(&frame);
//        }
//        //. Errror ????
//        //av_packet_unref(&packet);
//        packet.size = 0;
//    }
//
//    // Write the output video file trailer
//    av_write_trailer(outputFormatContext);
//
//    // Free resources
//    avformat_close_input(&formatContext);
//    avformat_free_context(outputFormatContext);
//    avcodec_free_context(&codecContext);
//
//    //. 
//    w_nSts = 0;
//    return w_nSts;
//}


static void encode(AVCodecContext *enc_ctx, AVFrame *frame, AVPacket *pkt,
                   FILE *outfile)
{
    int ret;

    /* send the frame to the encoder */
    if (frame)
        printf("Send frame %ld \n", frame->pts);

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

        printf("Write packet %ld\"PRId64\" (size=%d)\n", pkt->pts, pkt->size);
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
        fprintf(stderr, "Could not open codec: \n");
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
//.
int
MyFaceSuper_VideoProc_Ex(HANDLE p_h, std::string	p_strInVideo, std::string	p_strOutVideo)
{
    std::filesystem::path folderPath = std::filesystem::path(p_strInVideo).parent_path();

#ifdef WIN32
    std::string video_extract_path_in = folderPath.string() + "image_in\\";
    std::string video_extract_path_res = folderPath.string() + "image_result\\";
#else
    std::string video_extract_path_in = folderPath.string() + "image_in/";
    std::string video_extract_path_res = folderPath.string() + "image_result/";
#endif
    //. create directory
    std::filesystem::create_directory(video_extract_path_in);
    std::filesystem::create_directory(video_extract_path_res);

    //. extract image from video.
    std::string runpath = std::filesystem::current_path().string();
#ifdef WIN32
    std::string command = runpath + "\\ffmpeg.exe -i " + p_strInVideo + " " + video_extract_path_in + "frame%d.jpg";
#else
    std::string command = "ffmpeg -i " + p_strInVideo + " " + video_extract_path_in + "frame%d.jpg";
#endif
    system(command.c_str());

    cv::Mat  w_ImgMat_IN;
    cv::Mat  w_ImgMat_Out;
    int      loop = 0;

    for (auto& file_path : fs::directory_iterator(video_extract_path_in)) {

        std::string     w_filePath = file_path.path().u8string();
        std::string     w_fileName = file_path.path().filename().u8string();

        std::string     w_inFilePath = video_extract_path_in + w_fileName;
        std::string     w_outFilePath = video_extract_path_res + w_fileName;

        w_ImgMat_IN = cv::imread(w_inFilePath);
        OneFrameOfVideoProc(w_ImgMat_IN, w_ImgMat_Out);
        cv::imwrite(w_outFilePath, w_ImgMat_Out);

        loop++;
        std::cout << loop << "  frame OK" << std::endl;
    }

    //. combine 
#ifdef WIN32
    command = runpath + "\\ffmpeg.exe -framerate 30 -i " + video_extract_path_res + "frame%d.jpg" + " -c:v libx264 -r 30 -pix_fmt yuv420p " + p_strOutVideo;
#else
    command = "ffmpeg -framerate 30 -i " + video_extract_path_res + "frame%d.jpg" + " -c:v libx264 -r 30 -pix_fmt yuv420p " + p_strOutVideo;
#endif
    system(command.c_str());

    std::filesystem::remove_all(video_extract_path_in);
    std::filesystem::remove_all(video_extract_path_res);
    return 0;
}

//.
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
        av_opt_set(pOutCodecCtx->priv_data, "preset", "ultrafast", 0);

    /* open it */
    if (avcodec_open2(pOutCodecCtx, pOutCodec, NULL) < 0) {
        fprintf(stderr, "Could not open codec: \n");
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
//
//int
//MyEnhance_Video_File(std::string	p_strVideoFile_IN, std::string	p_strVideoFile_OUT)
//{
//    const char *filename, *codec_name;
//    const AVCodec *codec;
//    AVCodecContext *c= NULL;
//    int i, ret, x, y;
//    FILE *f;
//    AVFrame *frame;
//    AVPacket *pkt;
//    uint8_t endcode[] = { 0, 0, 1, 0xb7 };
//
//    filename = p_strVideoFile_OUT.c_str();
//    codec_name = "MPEG2VIDEO";
//
//    /* find the mpeg1video encoder */
//    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
//    if (!codec) {
//        fprintf(stderr, "Codec '%s' not found\n", codec_name);
//        exit(1);
//    }
//
//    c = avcodec_alloc_context3(codec);
//    if (!c) {
//        fprintf(stderr, "Could not allocate video codec context\n");
//        exit(1);
//    }
//
//    pkt = av_packet_alloc();
//    if (!pkt)
//        exit(1);
//
//    /* put sample parameters */
//    c->bit_rate = 400000;
//    /* resolution must be a multiple of two */
//    c->width = 352;
//    c->height = 288;
//    /* frames per second */
//    c->time_base.num = 1;
//    c->time_base.den = 25;
//
//        c->framerate.num = 25;
//    c->framerate.den = 1;
//
//    /* emit one intra frame every ten frames
//     * check frame pict_type before passing frame
//     * to encoder, if frame->pict_type is AV_PICTURE_TYPE_I
//     * then gop_size is ignored and the output of encoder
//     * will always be I frame irrespective to gop_size
//     */
//    c->gop_size = 10;
//    c->max_b_frames = 1;
//    c->pix_fmt = AV_PIX_FMT_YUV420P;
//
//    if (codec->id == AV_CODEC_ID_H264)
//        av_opt_set(c->priv_data, "preset", "slow", 0);
//
//    /* open it */
//    ret = avcodec_open2(c, codec, NULL);
//    if (ret < 0) {
//        fprintf(stderr, "Could not open codec: %d\n" , ret);
//        exit(1);
//    }
//
//    f = fopen(filename, "wb");
//    if (!f) {
//        fprintf(stderr, "Could not open %s\n", filename);
//        exit(1);
//    }
//
//    frame = av_frame_alloc();
//    if (!frame) {
//        fprintf(stderr, "Could not allocate video frame\n");
//        exit(1);
//    }
//    frame->format = c->pix_fmt;
//    frame->width  = c->width;
//    frame->height = c->height;
//
//    ret = av_frame_get_buffer(frame, 0);
//    if (ret < 0) {
//        fprintf(stderr, "Could not allocate the video frame data\n");
//        exit(1);
//    }
//
//    /* encode 1 second of video */
//    for (i = 0; i < 50; i++) {
//        fflush(stdout);
//
//        /* Make sure the frame data is writable.
//           On the first round, the frame is fresh from av_frame_get_buffer()
//           and therefore we know it is writable.
//           But on the next rounds, encode() will have called
//           avcodec_send_frame(), and the codec may have kept a reference to
//           the frame in its internal structures, that makes the frame
//           unwritable.
//           av_frame_make_writable() checks that and allocates a new buffer
//           for the frame only if necessary.
//         */
//        ret = av_frame_make_writable(frame);
//        if (ret < 0)
//            exit(1);
//
//        /* Prepare a dummy image.
//           In real code, this is where you would have your own logic for
//           filling the frame. FFmpeg does not care what you put in the
//           frame.
//         */
//        /* Y */
//        for (y = 0; y < c->height; y++) {
//            for (x = 0; x < c->width; x++) {
//                frame->data[0][y * frame->linesize[0] + x] = x + y + i * 3;
//            }
//        }
//
//        /* Cb and Cr */
//        for (y = 0; y < c->height/2; y++) {
//            for (x = 0; x < c->width/2; x++) {
//                frame->data[1][y * frame->linesize[1] + x] = 128 + y + i * 2;
//                frame->data[2][y * frame->linesize[2] + x] = 64 + x + i * 5;
//            }
//        }
//
//        frame->pts = i;
//
//        /* encode the image */
//        encode(c, frame, pkt, f);
//    }
//
//    /* flush the encoder */
//    encode(c, NULL, pkt, f);
//
//    /* Add sequence end code to have a real MPEG file.
//       It makes only sense because this tiny examples writes packets
//       directly. This is called "elementary stream" and only works for some
//       codecs. To create a valid file, you usually need to write packets
//       into a proper file format or protocol; see mux.c.
//     */
//    if (codec->id == AV_CODEC_ID_MPEG1VIDEO || codec->id == AV_CODEC_ID_MPEG2VIDEO)
//        fwrite(endcode, 1, sizeof(endcode), f);
//    fclose(f);
//
//    avcodec_free_context(&c);
//    av_frame_free(&frame);
//    av_packet_free(&pkt);
//
//    return 0;
//}
