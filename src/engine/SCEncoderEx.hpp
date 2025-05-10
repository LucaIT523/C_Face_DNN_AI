#ifndef __SCEncoderEx_HPP
#define __SCEncoderEx_HPP

#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <torch/script.h>
#include <iostream>
#include <math.h>
#include <random>

// Define Namespace
namespace nn = torch::nn;
using namespace std;

class NormStyleCodeImpl : public torch::nn::Module {
public:
    torch::Tensor forward(torch::Tensor x) {
        return x * torch::rsqrt(torch::mean(x.pow(2), 1, true) + 1e-8);
    }
};

TORCH_MODULE(NormStyleCode);

class ModulatedConv2dImpl : public torch::nn::Module {
public:
    ModulatedConv2dImpl(int in_channels, int out_channels, int kernel_size, int num_style_feat, bool demodulate = true, std::string sample_mode = "", double eps = 1e-8)
        : in_channels(in_channels),
        out_channels(out_channels),
        kernel_size(kernel_size),
        num_style_feat(num_style_feat),
        demodulate(demodulate),
        sample_mode(sample_mode),
        eps(eps) {
        modulation = register_module("modulation", torch::nn::Linear(num_style_feat, in_channels));
        weight = register_parameter("weight", torch::randn({ 1, out_channels, in_channels, kernel_size, kernel_size }) / sqrt(in_channels * kernel_size * kernel_size));
        padding = kernel_size / 2;
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor style) {
        auto b = x.size(0);
        auto c = x.size(1);
        auto h = x.size(2);
        auto w = x.size(3);

        auto style_mod = modulation->forward(style).view({ b, 1, c, 1, 1 });
        auto weight_mod = weight * style_mod;

        if (demodulate) {
            auto demod = torch::rsqrt(torch::sum(weight_mod.pow(2), { 2, 3, 4 }) + eps);
            weight_mod = weight_mod * demod.view({ b, out_channels, 1, 1, 1 });
        }

        weight_mod = weight_mod.view({ b * out_channels, c, kernel_size, kernel_size });

        if (sample_mode == "upsample") {
            x = torch::nn::UpsamplingBilinear2d(x, torch::IntArrayRef({ h * 2, w * 2 }));
        }
        else if (sample_mode == "downsample") {
            x = torch::upsample_bilinear2d(x, torch::IntArrayRef({ h / 2, w / 2 }));
        }

        b = x.size(0);
        c = x.size(1);
        h = x.size(2);
        w = x.size(3);

        x = x.view({ 1, b * c, h, w });
        auto out = torch::conv2d(x, weight_mod, torch::nn::Conv2dOptions(padding).groups(b));
        out = out.view({ b, out_channels, out.size(2), out.size(3) });
        return out;
    }

    torch::nn::Linear modulation;
    torch::Tensor weight;
    int in_channels, out_channels, kernel_size, num_style_feat, padding;
    bool demodulate;
    std::string sample_mode;
    double eps;
};

TORCH_MODULE(ModulatedConv2d);

class StyleConvImpl : public torch::nn::Module {
public:
    StyleConvImpl(int in_channels, int out_channels, int kernel_size, int num_style_feat, bool demodulate = true, std::string sample_mode = "")
        : in_channels(in_channels),
        out_channels(out_channels),
        kernel_size(kernel_size),
        num_style_feat(num_style_feat),
        demodulate(demodulate),
        sample_mode(sample_mode) {
        modulated_conv = register_module("modulated_conv", ModulatedConv2d(in_channels, out_channels, kernel_size, num_style_feat, demodulate, sample_mode));
        weight = register_parameter("weight", torch::zeros({ 1 }));
        bias = register_parameter("bias", torch::zeros({ 1, out_channels, 1, 1 }));
        activate = torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor style, torch::Tensor noise = nullptr) {
        auto out = modulated_conv->forward(x, style) * sqrt(2.0);

        if (noise == nullptr) {
            auto b = out.size(0);
            auto h = out.size(2);
            auto w = out.size(3);
            noise = torch::zeros({ b, 1, h, w }).normal_();
        }

        out = out + weight * noise;
        out = out + bias;
        out = activate->forward(out);
        return out;
    }

    ModulatedConv2d modulated_conv;
    torch::Tensor weight, bias;
    int in_channels, out_channels, kernel_size, num_style_feat;
    std::string sample_mode;
    torch::nn::LeakyReLU activate;
};

TORCH_MODULE(StyleConv);

class ToRGBImpl : public torch::nn::Module {
public:
    ToRGBImpl(int in_channels, int num_style_feat, bool upsample = true)
        : in_channels(in_channels),
        num_style_feat(num_style_feat),
        upsample(upsample) {
        modulated_conv = register_module("modulated_conv", ModulatedConv2d(in_channels, 3, 1, num_style_feat, false, ""));
        bias = register_parameter("bias", torch::zeros({ 1, 3, 1, 1 }));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor style, torch::Tensor skip = nullptr) {
        auto out = modulated_conv->forward(x, style) + bias;

        if (skip != nullptr) {
            if (upsample) {
                skip = torch::upsample_bilinear2d(skip, torch::IntArrayRef({ skip.size(2) * 2, skip.size(3) * 2 }));
            }
            out = out + skip;
        }

        return out;
    }

    ModulatedConv2d modulated_conv;
    torch::Tensor bias;
    int in_channels, num_style_feat;
    bool upsample;
};

TORCH_MODULE(ToRGB);

class ConstantInputImpl : public torch::nn::Module {
public:
    ConstantInputImpl(int num_channel, int size)
        : num_channel(num_channel),
        size(size) {
        weight = register_parameter("weight", torch::randn({ 1, num_channel, size, size }));
    }

    torch::Tensor forward(int batch) {
        auto out = weight.repeat({ batch, 1, 1, 1 });
        return out;
    }

    torch::Tensor weight;
    int num_channel, size;
};

TORCH_MODULE(ConstantInput);

class StyleGAN2GeneratorCleanImpl : public torch::nn::Module {
public:
    StyleGAN2GeneratorCleanImpl(int out_size, int num_style_feat = 512, int num_mlp = 8, int channel_multiplier = 2, double narrow = 1.0)
        : out_size(out_size),
        num_style_feat(num_style_feat),
        num_mlp(num_mlp),
        channel_multiplier(channel_multiplier),
        narrow(narrow) {
        style_mlp = register_module("style_mlp", build_style_mlp(num_style_feat, num_mlp));
        constant_input = register_module("constant_input", ConstantInput(channels["4"], 4));
        style_conv1 = register_module("style_conv1", StyleConv(channels["4"], channels["4"], 3, num_style_feat, true, ""));
        to_rgb1 = register_module("to_rgb1", ToRGB(channels["4"], num_style_feat, false));
        log_size = log2(out_size);
        num_layers = (log_size - 2) * 2 + 1;
        num_latent = log_size * 2 - 2;
        style_convs = register_module("style_convs", build_style_convs());
        to_rgbs = register_module("to_rgbs", build_to_rgbs());
        noises = register_module("noises", build_noises());
    }

    torch::nn::Sequential build_style_mlp(int num_style_feat, int num_mlp) {
        torch::nn::Sequential layers;
        layers->push_back(NormStyleCode());
        for (int i = 0; i < num_mlp; i++) {
            layers->push_back(torch::nn::Linear(num_style_feat, num_style_feat));
            layers->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
        }
        return layers;
    }

    torch::nn::ModuleList build_style_convs() {
        torch::nn::ModuleList layers;
        int in_channels = channels["4"];
        for (int i = 3; i <= log_size; i++) {
            int out_channels = channels[std::to_string(pow(2, i))];
            layers->push_back(StyleConv(in_channels, out_channels, 3, num_style_feat, true, "upsample"));
            layers->push_back(StyleConv(out_channels, out_channels, 3, num_style_feat, true, ""));
            in_channels = out_channels;
        }
        return layers;
    }

    torch::nn::ModuleList build_to_rgbs() {
        torch::nn::ModuleList layers;
        int in_channels = channels["4"];
        for (int i = 3; i <= log_size; i++) {
            int out_channels = channels[std::to_string(pow(2, i))];
            layers->push_back(ToRGB(out_channels, num_style_feat, true));
        }
        return layers;
    }

    torch::nn::Module build_noises() {
        torch::nn::Module noises;
        for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
            int resolution = pow(2, (layer_idx + 5) / 2);
            torch::Tensor noise = torch::randn({ 1, 1, resolution, resolution });
            noises->register_buffer("noise" + std::to_string(layer_idx), noise);
        }
        return noises;
    }

    torch::Tensor make_noise() {
        std::vector<torch::Tensor> noise;
        for (int i = 0; i < num_layers; i++) {
            noise.push_back(noises->get_buffer("noise" + std::to_string(i)));
        }
        return noise;
    }

    torch::Tensor forward(torch::Tensor styles, bool input_is_latent = false, torch::Tensor noise = nullptr, bool randomize_noise = true, double truncation = 1.0, torch::Tensor truncation_latent = nullptr, int inject_index = -1, bool return_latents = false) {
        if (!input_is_latent) {
            styles = style_mlp->forward(styles);
        }

        if (noise == nullptr) {
            if (randomize_noise) {
                noise = make_noise();
            }
            else {
                noise = noises;
            }
        }

        if (truncation < 1.0) {
            if (truncation_latent == nullptr) {
                truncation_latent = style_mlp->forward(torch::randn({ 1, num_style_feat }));
            }
            styles = truncation_latent + truncation * (styles - truncation_latent);
        }

        torch::Tensor latent;
        if (styles.size(0) == 1) {
            if (styles.dim() < 3) {
                latent = styles.unsqueeze(1).repeat({ 1, inject_index, 1 });
            }
            else {
                latent = styles;
            }
        }
        else if (styles.size(0) == 2) {
            if (inject_index == -1) {
                inject_index = torch::randint(1, num_latent - 1, { 1 }).item<int>();
            }
            torch::Tensor latent1 = styles[0].unsqueeze(1).repeat({ 1, inject_index, 1 });
            torch::Tensor latent2 = styles[1].unsqueeze(1).repeat({ 1, num_latent - inject_index, 1 });
            latent = torch::cat({ latent1, latent2 }, 1);
        }

        torch::Tensor out = constant_input->forward(latent.size(0));
        out = style_conv1->forward(out, latent[0], noise[0]);
        torch::Tensor skip = to_rgb1->forward(out, latent[1]);

        int i = 1;
        for (auto& conv1 : style_convs) {
            auto& conv2 = style_convs[i];
            auto& noise1 = noise[i];
            auto& noise2 = noise[i + 1];
            auto& to_rgb = to_rgbs[i + 2];

            out = conv1->forward(out, latent[i], noise1);
            out = conv2->forward(out, latent[i + 1], noise2);
            skip = to_rgb->forward(out, latent[i + 2], skip);

            i += 2;
        }

        torch::Tensor image = skip;
        if (return_latents) {
            return { image, latent };
        }
        else {
            return { image, torch::Tensor() };
        }
    }

    int out_size, num_style_feat, num_mlp, log_size, num_layers, num_latent;
    double narrow;
    std::map<std::string, int> channels = {
        {"4", 512},
        {"8", 512},
        {"16", 512},
        {"32", 512},
        {"64", 256},
        {"128", 128},
        {"256", 64},
        {"512", 32},
        {"1024", 16}
    };
    torch::nn::Sequential style_mlp;
    ConstantInput constant_input;
    StyleConv style_conv1;
    ToRGB to_rgb1;
    torch::nn::ModuleList style_convs, to_rgbs;
    torch::nn::Module noises;
};

TORCH_MODULE(StyleGAN2GeneratorClean);

















#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <torch/script.h>
#include <iostream>
#include <math.h>
#include <random>

namespace basicsr {
    namespace archs {

        class NormStyleCodeImpl : public torch::nn::Module {
        public:
            torch::Tensor forward(torch::Tensor x) {
                return x * torch::rsqrt(torch::mean(x.pow(2), 1, true) + 1e-8);
            }
        };

        TORCH_MODULE(NormStyleCode);

        class ModulatedConv2dImpl : public torch::nn::Module {
        public:
            ModulatedConv2dImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t num_style_feat,
                bool demodulate = true, std::string sample_mode = nullptr, double eps = 1e-8)
                : in_channels(in_channels),
                out_channels(out_channels),
                kernel_size(kernel_size),
                num_style_feat(num_style_feat),
                demodulate(demodulate),
                sample_mode(sample_mode),
                eps(eps),
                modulation(torch::nn::Linear(num_style_feat, in_channels)),
                weight(torch::randn({ 1, out_channels, in_channels, kernel_size, kernel_size }) /
                    sqrt(in_channels * kernel_size * kernel_size)),
                padding(kernel_size / 2) {
                register_module("modulation", modulation);
                register_parameter("weight", weight);
            }

            torch::Tensor forward(torch::Tensor x, torch::Tensor style) {
                auto b = x.size(0);
                auto c = x.size(1);
                auto h = x.size(2);
                auto w = x.size(3);

                auto style_mod = modulation->forward(style).view({ b, 1, c, 1, 1 });
                auto weight_mod = weight * style_mod;

                if (demodulate) {
                    auto demod = torch::rsqrt(torch::sum(weight_mod.pow(2), { 2, 3, 4 }) + eps);
                    weight_mod = weight_mod * demod.view({ b, out_channels, 1, 1, 1 });
                }

                weight_mod = weight_mod.view({ b * out_channels, c, kernel_size, kernel_size });

                if (sample_mode == "upsample") {
                    x = torch::nn::functional::interpolate(x, torch::nn::functional::InterpolateFuncOptions()
                        .scale_factor(2).mode(torch::kModeBilinear).align_corners(false));
                }
                else if (sample_mode == "downsample") {
                    x = torch::nn::functional::interpolate(x, torch::nn::functional::InterpolateFuncOptions()
                        .scale_factor(0.5).mode(torch::kModeBilinear).align_corners(false));
                }

                b = x.size(0);
                c = x.size(1);
                h = x.size(2);
                w = x.size(3);

                x = x.view({ 1, b * c, h, w });
                auto out = torch::nn::functional::conv2d(x, weight_mod, torch::nn::functional::Conv2dFuncOptions()
                    .padding(padding).groups(b));
                out = out.view({ b, out_channels, out.size(2), out.size(3) });

                return out;
            }

            int64_t in_channels;
            int64_t out_channels;
            int64_t kernel_size;
            int64_t num_style_feat;
            bool demodulate;
            std::string sample_mode;
            double eps;

            torch::nn::Linear modulation;
            torch::Tensor weight;
            int64_t padding;
        };

        TORCH_MODULE(ModulatedConv2d);

        class StyleConvImpl : public torch::nn::Module {
        public:
            StyleConvImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t num_style_feat,
                bool demodulate = true, std::string sample_mode = nullptr)
                : modulated_conv(ModulatedConv2d(in_channels, out_channels, kernel_size, num_style_feat, demodulate, sample_mode)),
                weight(torch::zeros({ 1 })),
                bias(torch::zeros({ 1, out_channels, 1, 1 })),
                activate(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true))) {
                register_module("modulated_conv", modulated_conv);
                register_parameter("weight", weight);
                register_parameter("bias", bias);
            }

            torch::Tensor forward(torch::Tensor x, torch::Tensor style, torch::Tensor noise = nullptr) {
                auto out = modulated_conv->forward(x, style) * sqrt(2.0);

                if (noise == nullptr) {
                    auto b = out.size(0);
                    auto h = out.size(2);
                    auto w = out.size(3);
                    noise = torch::empty({ b, 1, h, w }).normal_();
                }

                out = out + weight * noise;
                out = out + bias;
                out = activate->forward(out);

                return out;
            }

            ModulatedConv2d modulated_conv;
            torch::Tensor weight;
            torch::Tensor bias;
            torch::nn::LeakyReLU activate;
        };

        TORCH_MODULE(StyleConv);

        class ToRGBImpl : public torch::nn::Module {
        public:
            ToRGBImpl(int64_t in_channels, int64_t num_style_feat, bool upsample = true)
                : upsample(upsample),
                modulated_conv(ModulatedConv2d(in_channels, 3, 1, num_style_feat, false, nullptr)),
                bias(torch::zeros({ 1, 3, 1, 1 })) {
                register_module("modulated_conv", modulated_conv);
                register_parameter("bias", bias);
            }

            torch::Tensor forward(torch::Tensor x, torch::Tensor style, torch::Tensor skip = nullptr) {
                auto out = modulated_conv->forward(x, style) + bias;

                if (skip != nullptr) {
                    if (upsample) {
                        skip = torch::nn::functional::interpolate(skip, torch::nn::functional::InterpolateFuncOptions()
                            .scale_factor(2).mode(torch::kModeBilinear).align_corners(false));
                    }
                    out = out + skip;
                }

                return out;
            }

            bool upsample;
            ModulatedConv2d modulated_conv;
            torch::Tensor bias;
        };

        TORCH_MODULE(ToRGB);

        class ConstantInputImpl : public torch::nn::Module {
        public:
            ConstantInputImpl(int64_t num_channel, int64_t size)
                : weight(torch::randn({ 1, num_channel, size, size })) {
                register_parameter("weight", weight);
            }

            torch::Tensor forward(int64_t batch) {
                auto out = weight.repeat({ batch, 1, 1, 1 });
                return out;
            }

            torch::Tensor weight;
        };

        TORCH_MODULE(ConstantInput);

        class StyleGAN2GeneratorCleanImpl : public torch::nn::Module {
        public:
            StyleGAN2GeneratorCleanImpl(int64_t out_size, int64_t num_style_feat = 512, int64_t num_mlp = 8,
                int64_t channel_multiplier = 2, double narrow = 1.0)
                : num_style_feat(num_style_feat),
                style_mlp(NormStyleCode()),
                constant_input(ConstantInput(int(512 * narrow), 4)),
                style_conv1(StyleConv(int(512 * narrow), int(512 * narrow), 3, num_style_feat)),
                to_rgb1(ToRGB(int(512 * narrow), num_style_feat, false)),
                log_size(log2(out_size)),
                num_layers((log_size - 2) * 2 + 1),
                num_latent(log_size * 2 - 2) {
                register_module("style_mlp", style_mlp);
                register_module("constant_input", constant_input);
                register_module("style_conv1", style_conv1);
                register_module("to_rgb1", to_rgb1);

                std::unordered_map<int64_t, int64_t> channels = {
                    { 4, int(512 * narrow) },
                    { 8, int(512 * narrow) },
                    { 16, int(512 * narrow) },
                    { 32, int(512 * narrow) },
                    { 64, int(256 * channel_multiplier * narrow) },
                    { 128, int(128 * channel_multiplier * narrow) },
                    { 256, int(64 * channel_multiplier * narrow) },
                    { 512, int(32 * channel_multiplier * narrow) },
                    { 1024, int(16 * channel_multiplier * narrow) }
                };
                this->channels = channels;

                for (int64_t layer_idx = 0; layer_idx < num_layers; layer_idx++) {
                    int64_t resolution = pow(2, (layer_idx + 5) / 2);
                    std::vector<int64_t> shape = { 1, 1, resolution, resolution };
                    auto noise = torch::randn(shape);
                    register_buffer("noise" + std::to_string(layer_idx), noise);
                }

                for (int64_t i = 3; i <= log_size; i++) {
                    int64_t out_channels = channels[pow(2, i)];
                    style_convs.push_back(StyleConv(channels[pow(2, i - 1)], out_channels, 3, num_style_feat, true, "upsample"));
                    style_convs.push_back(StyleConv(out_channels, out_channels, 3, num_style_feat, true, nullptr));
                    to_rgbs.push_back(ToRGB(out_channels, num_style_feat, true));
                }
            }

            std::vector<torch::Tensor> make_noise() {
                std::vector<torch::Tensor> noises;
                noises.push_back(torch::randn({ 1, 1, 4, 4 }));

                for (int64_t i = 3; i <= log_size; i++) {
                    noises.push_back(torch::randn({ 1, 1, pow(2, i), pow(2, i) }));
                    noises.push_back(torch::randn({ 1, 1, pow(2, i), pow(2, i) }));
                }

                return noises;
            }

            torch::Tensor get_latent(torch::Tensor x) {
                return style_mlp->forward(x);
            }

            torch::Tensor mean_latent(int64_t num_latent) {
                auto latent_in = torch::randn({ num_latent, num_style_feat });
                auto latent = style_mlp->forward(latent_in).mean(0, true);
                return latent;
            }

            std::vector<torch::Tensor> forward(std::vector<torch::Tensor> styles, bool input_is_latent = false,
                torch::Tensor noise = nullptr, bool randomize_noise = true, double truncation = 1,
                torch::Tensor truncation_latent = nullptr, int64_t inject_index = nullptr, bool return_latents = false) {
                if (!input_is_latent) {
                    for (int64_t i = 0; i < styles.size(); i++) {
                        styles[i] = style_mlp->forward(styles[i]);
                    }
                }

                if (noise == nullptr) {
                    if (randomize_noise) {
                        noise = std::vector<torch::Tensor>(num_layers, nullptr);
                    }
                    else {
                        noise = std::vector<torch::Tensor>();
                        for (int64_t layer_idx = 0; layer_idx < num_layers; layer_idx++) {
                            noise.push_back(getattr("noise" + std::to_string(layer_idx)));
                        }
                    }
                }

                if (truncation < 1) {
                    std::vector<torch::Tensor> style_truncation;
                    for (int64_t i = 0; i < styles.size(); i++) {
                        style_truncation.push_back(truncation_latent + truncation * (styles[i] - truncation_latent));
                    }
                    styles = style_truncation;
                }

                torch::Tensor latent;
                if (styles.size() == 1) {
                    inject_index = num_latent;

                    if (styles[0].dim() < 3) {
                        latent = styles[0].unsqueeze(1).repeat({ 1, inject_index, 1 });
                    }
                    else {
                        latent = styles[0];
                    }
                }
                else if (styles.size() == 2) {
                    if (inject_index == nullptr) {
                        inject_index = std::rand() % (num_latent - 1) + 1;
                    }

                    auto latent1 = styles[0].unsqueeze(1).repeat({ 1, inject_index, 1 });
                    auto latent2 = styles[1].unsqueeze(1).repeat({ 1, num_latent - inject_index, 1 });
                    latent = torch::cat({ latent1, latent2 }, 1);
                }

                torch::Tensor out = constant_input->forward(latent.size(0));
                out = style_conv1->forward(out, latent.select(1, 0), noise[0]);
                torch::Tensor skip = to_rgb1->forward(out, latent.select(1, 1));

                int64_t i = 1;
                for (int64_t j = 0; j < style_convs.size(); j += 2) {
                    out = style_convs[j]->forward(out, latent.select(1, i), noise[j + 1]);
                    out = style_convs[j + 1]->forward(out, latent.select(1, i + 1), noise[j + 2]);
                    skip = to_rgbs[j / 2]->forward(out, latent.select(1, i + 2), skip);
                    i += 2;
                }

                torch::Tensor image = skip;

                if (return_latents) {
                    return { image, latent };
                }
                else {
                    return { image };
                }
            }

            int64_t num_style_feat;
            NormStyleCode style_mlp;
            ConstantInput constant_input;
            StyleConv style_conv1;
            ToRGB to_rgb1;
            int64_t log_size;
            int64_t num_layers;
            int64_t num_latent;
            std::unordered_map<int64_t, int64_t> channels;
            std::vector<StyleConv> style_convs;
            std::vector<ToRGB> to_rgbs;
        };

        TORCH_MODULE(StyleGAN2GeneratorClean);

    } // namespace archs
} // namespace basicsr

Please note that this is a direct translation of the codeand may require additional modifications or adjustments based on your specific use case.
Was the last answer useful ?
Ask any technical question... (Ctrl + Enter for newline)
Send
Also available for :
    VSCode
    JetBrains
    CLI
    bito.ai

#endif // __SCEncoderEx_HPP