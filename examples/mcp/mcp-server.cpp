#include "common.h"
#include "common-whisper.h"

#include "whisper.h"
#include "json.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <iostream>

using json = nlohmann::ordered_json;

namespace {

const std::string json_format   = "json";

struct mcp_params
{
    bool ffmpeg_converter = false;
};

struct whisper_params {
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors  = 1;
    int32_t offset_t_ms   = 0;
    int32_t offset_n      = 0;
    int32_t duration_ms   = 0;
    int32_t progress_step = 5;
    int32_t max_context   = -1;
    int32_t max_len       = 0;
    int32_t best_of       = 2;
    int32_t beam_size     = -1;
    int32_t audio_ctx     = 0;

    float word_thold      =  0.01f;
    float entropy_thold   =  2.40f;
    float logprob_thold   = -1.00f;
    float temperature     =  0.00f;
    float temperature_inc =  0.20f;
    float no_speech_thold = 0.6f;

    bool debug_mode      = false;
    bool translate       = false;
    bool detect_language = false;
    bool diarize         = false;
    bool tinydiarize     = false;
    bool split_on_word   = false;
    bool no_fallback     = false;
    bool print_special   = false;
    bool print_colors    = false;
    bool print_realtime  = false;
    bool print_progress  = false;
    bool no_timestamps   = false;
    bool use_gpu         = true;
    bool flash_attn      = false;
    bool suppress_nst    = false;
    bool no_context      = false;

    std::string language        = "en";
    std::string prompt          = "";
    std::string model           = "models/ggml-base.en.bin";

    std::string response_format     = json_format;

    std::string openvino_encode_device = "CPU";

    std::string dtw = "";
};

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params, const mcp_params & mparams) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] \n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,        --help              [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,      --threads N         [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "  -p N,      --processors N      [%-7d] number of processors to use during computation\n", params.n_processors);
    fprintf(stderr, "  -ot N,     --offset-t N        [%-7d] time offset in milliseconds\n",                    params.offset_t_ms);
    fprintf(stderr, "  -on N,     --offset-n N        [%-7d] segment index offset\n",                           params.offset_n);
    fprintf(stderr, "  -d  N,     --duration N        [%-7d] duration of audio to process in milliseconds\n",   params.duration_ms);
    fprintf(stderr, "  -mc N,     --max-context N     [%-7d] maximum number of text context tokens to store\n", params.max_context);
    fprintf(stderr, "  -ml N,     --max-len N         [%-7d] maximum segment length in characters\n",           params.max_len);
    fprintf(stderr, "  -sow,      --split-on-word     [%-7s] split on word rather than on token\n",             params.split_on_word ? "true" : "false");
    fprintf(stderr, "  -bo N,     --best-of N         [%-7d] number of best candidates to keep\n",              params.best_of);
    fprintf(stderr, "  -bs N,     --beam-size N       [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -ac N,     --audio-ctx N       [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -wt N,     --word-thold N      [%-7.2f] word timestamp probability threshold\n",         params.word_thold);
    fprintf(stderr, "  -et N,     --entropy-thold N   [%-7.2f] entropy threshold for decoder fail\n",           params.entropy_thold);
    fprintf(stderr, "  -lpt N,    --logprob-thold N   [%-7.2f] log probability threshold for decoder fail\n",   params.logprob_thold);
    fprintf(stderr, "  -debug,    --debug-mode        [%-7s] enable debug mode (eg. dump log_mel)\n",           params.debug_mode ? "true" : "false");
    fprintf(stderr, "  -tr,       --translate         [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -di,       --diarize           [%-7s] stereo audio diarization\n",                       params.diarize ? "true" : "false");
    fprintf(stderr, "  -tdrz,     --tinydiarize       [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -nf,       --no-fallback       [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -nt,       --no-timestamps     [%-7s] do not print timestamps\n",                        params.no_timestamps ? "true" : "false");
    fprintf(stderr, "  -l LANG,   --language LANG     [%-7s] spoken language ('auto' for auto-detect)\n",       params.language.c_str());
    fprintf(stderr, "  -dl,       --detect-language   [%-7s] exit after automatically detecting language\n",    params.detect_language ? "true" : "false");
    fprintf(stderr, "             --prompt PROMPT     [%-7s] initial prompt\n",                                 params.prompt.c_str());
    fprintf(stderr, "  -m FNAME,  --model FNAME       [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -oved D,   --ov-e-device DNAME [%-7s] the OpenVINO device used for encode inference\n",  params.openvino_encode_device.c_str());
    // mcp params
    fprintf(stderr, "  --convert,                     [%-7s] Convert audio to WAV, requires ffmpeg on the server\n", mparams.ffmpeg_converter ? "true" : "false");

    fprintf(stderr, "  -sns,      --suppress-nst      [%-7s] suppress non-speech tokens\n", params.suppress_nst ? "true" : "false");
    fprintf(stderr, "  -nth N,    --no-speech-thold N [%-7.2f] no speech threshold\n",   params.no_speech_thold);
    fprintf(stderr, "  -nc,       --no-context        [%-7s] do not use previous audio context\n", params.no_context ? "true" : "false");
    fprintf(stderr, "  -ng,       --no-gpu            [%-7s] do not use gpu\n", params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,       --flash-attn        [%-7s] flash attention\n", params.flash_attn ? "true" : "false");
    fprintf(stderr, "\n");
}

bool whisper_params_parse(int argc, char ** argv, whisper_params & params, mcp_params & mparams) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params, mparams);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")         { params.n_threads       = std::stoi(argv[++i]); }
        else if (arg == "-p"    || arg == "--processors")      { params.n_processors    = std::stoi(argv[++i]); }
        else if (arg == "-ot"   || arg == "--offset-t")        { params.offset_t_ms     = std::stoi(argv[++i]); }
        else if (arg == "-on"   || arg == "--offset-n")        { params.offset_n        = std::stoi(argv[++i]); }
        else if (arg == "-d"    || arg == "--duration")        { params.duration_ms     = std::stoi(argv[++i]); }
        else if (arg == "-mc"   || arg == "--max-context")     { params.max_context     = std::stoi(argv[++i]); }
        else if (arg == "-ml"   || arg == "--max-len")         { params.max_len         = std::stoi(argv[++i]); }
        else if (arg == "-bo"   || arg == "--best-of")         { params.best_of         = std::stoi(argv[++i]); }
        else if (arg == "-bs"   || arg == "--beam-size")       { params.beam_size       = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")       { params.audio_ctx       = std::stoi(argv[++i]); }
        else if (arg == "-wt"   || arg == "--word-thold")      { params.word_thold      = std::stof(argv[++i]); }
        else if (arg == "-et"   || arg == "--entropy-thold")   { params.entropy_thold   = std::stof(argv[++i]); }
        else if (arg == "-lpt"  || arg == "--logprob-thold")   { params.logprob_thold   = std::stof(argv[++i]); }
        else if (arg == "-debug"|| arg == "--debug-mode")      { params.debug_mode      = true; }
        else if (arg == "-tr"   || arg == "--translate")       { params.translate       = true; }
        else if (arg == "-di"   || arg == "--diarize")         { params.diarize         = true; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")     { params.tinydiarize     = true; }
        else if (arg == "-sow"  || arg == "--split-on-word")   { params.split_on_word   = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")     { params.no_fallback     = true; }
        else if (arg == "-nt"   || arg == "--no-timestamps")   { params.no_timestamps   = true; }
        else if (arg == "-l"    || arg == "--language")        { params.language        = argv[++i]; }
        else if (arg == "-dl"   || arg == "--detect-language") { params.detect_language = true; }
        else if (                  arg == "--prompt")          { params.prompt          = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")           { params.model           = argv[++i]; }
        else if (arg == "-oved" || arg == "--ov-e-device")     { params.openvino_encode_device = argv[++i]; }
        else if (arg == "-dtw"  || arg == "--dtw")             { params.dtw             = argv[++i]; }
        else if (arg == "-ng"   || arg == "--no-gpu")          { params.use_gpu         = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")      { params.flash_attn      = true; }
        else if (arg == "-sns"  || arg == "--suppress-nst")    { params.suppress_nst    = true; }
        else if (arg == "-nth"  || arg == "--no-speech-thold") { params.no_speech_thold = std::stof(argv[++i]); }
        else if (arg == "-nc"   || arg == "--no-context")      { params.no_context      = true; }

        // mcp server params
        else if (                  arg == "--convert")         { mparams.ffmpeg_converter     = true; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params, mparams);
            exit(0);
        }
        GGML_UNUSED(mparams);
    }

    return true;
}

struct whisper_print_user_data {
    const whisper_params * params;

    const std::vector<std::vector<float>> * pcmf32s;
    int progress_prev;
};

void check_ffmpeg_availibility() {
    int result = system("ffmpeg -version");

    if (result == 0) {
        printf("ffmpeg is available.\n");
    } else {
        printf("ffmpeg is not available.\n");
        exit(0);
    }
}

bool convert_to_wav(const std::string & temp_filename, std::string & error_resp) {
    std::ostringstream cmd_stream;
    std::string converted_filename_temp = temp_filename + "_temp.wav";
    cmd_stream << "ffmpeg -i \"" << temp_filename << "\" -y -ar 16000 -ac 1 -c:a pcm_s16le \"" << converted_filename_temp << "\" 2>&1";
    std::string cmd = cmd_stream.str();

    int status = std::system(cmd.c_str());
    if (status != 0) {
        error_resp = "{\"error\":\"FFmpeg conversion failed.\"}";
        return false;
    }

    // Remove the original file
    if (remove(temp_filename.c_str()) != 0) {
        error_resp = "{\"error\":\"Failed to remove the original file.\"}";
        return false;
    }

    // Rename the temporary file to match the original filename
    if (rename(converted_filename_temp.c_str(), temp_filename.c_str()) != 0) {
        error_resp = "{\"error\":\"Failed to rename the temporary file.\"}";
        return false;
    }
    return true;
}

}  // namespace

class WhisperMCPServer {
private:
    struct whisper_context * ctx;
    std::string              model_path;
    bool                     model_loaded;
    struct mcp_params        mparams;
    struct whisper_params    wparams;

public:
    WhisperMCPServer(const mcp_params mparams,
            const whisper_params & wparams,
            const std::string & model = "models/ggml-base.en.bin")
        : ctx(nullptr), model_path(model), model_loaded(false), mparams(mparams), wparams(wparams) {}

    ~WhisperMCPServer() {
        if (ctx) {
            whisper_free(ctx);
        }
    }

    bool load_model() {
        if (model_loaded) {
            return true;
        }

        fprintf(stderr, "%s: Loading whisper model from: %s\n", __func__, model_path.c_str());

        struct whisper_context_params cparams = whisper_context_default_params();

        ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
        if (!ctx) {
            fprintf(stderr, "%s: Failed to load model: %s", __func__, model_path.c_str());

            return false;
        }

        model_loaded = true;
        fprintf(stderr, "%s: Model loaded successfully!\n", __func__);
        return true;
    }

    std::string transcribe_file(const std::string & filepath, const std::string & language = "auto", bool translate = false) {
        if (!model_loaded) {
            throw std::runtime_error("Model not loaded");
        }

        struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

        if (language != "auto" && whisper_lang_id(language.c_str()) == -1) {
            fprintf(stderr, "error: unknown language '%s'\n", wparams.language);
            throw std::runtime_error("Unknown language: " + language);
        }
        if (language != "auto") {
            wparams.language = language.c_str();
        } else {
            wparams.language = "auto";
        }

        wparams.translate = translate;

        // Disable printing to stdout/stderr during processing
        wparams.print_progress = false;
        wparams.print_timestamps = false;

        std::vector<float> pcmf32;
        if (!load_audio_file(filepath, pcmf32)) {
            throw std::runtime_error("Failed to load audio file: " + filepath);
        }

        if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
            throw std::runtime_error("Whisper inference failed");
        }

        // Extract transcription
        std::string result;
        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) {
            const char* text = whisper_full_get_segment_text(ctx, i);
            result += text;
        }

        return result;
    }

    bool load_audio_file(const std::string & fname_inp, std::vector<float> & pcmf32) {
        fprintf(stderr, "%s: Loading audio file: %s", __func__, fname_inp.c_str());
        std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

        if (!::read_audio_data(fname_inp, pcmf32, pcmf32s, wparams.diarize)) {
            fprintf(stderr, "error: failed to read audio file '%s'\n", fname_inp.c_str());
            return false;
        }

        fprintf(stderr, "Successfully loaded %s\n", fname_inp.c_str());
        return true;
    }

    void send_response(const json & response) {
        printf("%s\n", response.dump().c_str());
        fflush(stdout);
    }

    // Send JSON-RPC success response
    void send_result(const json & id, const json & result) {
        json response = {
            {"jsonrpc", "2.0"},
            {"result", result}
        };

        // Only add id if it's not null
        if (!id.is_null()) {
            response["id"] = id;
        }

        send_response(response);
    }

    // Send JSON-RPC error response
    void send_error(const json & id, int code, const std::string & message) {
        json response = {
            {"jsonrpc", "2.0"},
            {"id", id},
            {"error", {
                {"code", code},
                {"message", message}
            }}
        };
        send_response(response);
    }

    void handle_initialize(const json & id, const json & params) {
        fprintf(stderr, "Model path: %s\n", model_path.c_str());
        if (!load_model()) {
            send_error(id, -32603, "Failed to load whisper model");
            return;
        }

        // Send proper initialize response
        json result = {
            {"protocolVersion", "2024-11-05"},
            {"capabilities", {
                {"tools", json::object()} // This server supports tools with a default {} tool capabilities
            }},
            {"serverInfo", {
                {"name", "whisper-mcp-server"},
                {"version", "1.0.0"}
            }}
        };

        send_result(id, result);
    }

    void handle_list_tools(const json & id) {
        fprintf(stderr, "Listing tools...\n");
        json result = {
            {"tools", json::array({
                {
                    {"name", "transcribe"},
                    {"description", "Transcribe audio file using persistent whisper.cpp model"},
                    {"inputSchema", {
                        {"type", "object"},
                        {"properties", {
                            {"file", {
                                {"type", "string"},
                                {"description", "Path to audio file"}
                            }},
                            {"language", {
                                {"type", "string"},
                                {"description", "Language code (optional, auto-detect if not specified)"},
                                {"default", "auto"}
                            }},
                            {"translate", {
                                {"type", "boolean"},
                                {"description", "Translate to English"},
                                {"default", false}
                            }}
                        }},
                        {"required", json::array({"file"})}
                    }}
                },
                {
                    {"name", "model_info"},
                    {"description", "Get information about loaded model"},
                    {"inputSchema", {
                        {"type", "object"},
                        {"properties", json::object()}
                    }}
                }
            })}
        };
        send_result(id, result);
    }

    void handle_transcribe(const json & id, const json & arguments) {
        try {
            if (!arguments.contains("file")) {
                send_error(id, -32602, "Missing required parameter: file");
                return;
            }

            std::string filePath = arguments["file"];
            std::string language = arguments.value("language", "auto");
            bool translate = arguments.value("translate", false);

            std::string transcription = transcribe_file(filePath, language, translate);

            json result = {
                {"content", json::array({
                    {
                        {"type", "text"},
                        {"text", transcription}
                    }
                })}
            };

            send_result(id, result);

        } catch (const std::exception & e) {
            send_error(id, -32603, std::string("Transcription failed: ") + e.what());
        }
    }

    void handle_model_info(const json & id) {
        if (!model_loaded) {
            send_error(id, -32603, "No model loaded");
            return;
        }

        // Get model information
        json modelInfo = {
            {"model_path", model_path},
            {"model_loaded", model_loaded},
            {"vocab_size", whisper_n_vocab(ctx)},
            {"n_text_ctx", whisper_n_text_ctx(ctx)},
            {"n_audio_ctx", whisper_n_audio_ctx(ctx)},
            {"is_multilingual", whisper_is_multilingual(ctx)}
        };

        json result = {
            {"content", json::array({
                {
                    {"type", "text"},
                    {"text", "Model Information:\n" + modelInfo.dump(2)}
                }
            })}
        };

        send_result(id, result);
    }

    void handle_tool_call(const json & id, const json & params) {
        if (!params.contains("name")) {
            send_error(id, -32602, "Missing tool name");
            return;
        }

        std::string toolName = params["name"];
        json arguments = params.value("arguments", json::object());

        if (toolName == "transcribe") {
            handle_transcribe(id, arguments);
        } else if (toolName == "model_info") {
            handle_model_info(id);
        } else {
            send_error(id, -32601, "Unknown tool: " + toolName);
        }
    }

    void run() {
        fprintf(stderr, "MCP Server starting...\n");
        fprintf(stderr, "Model path: %s\n", model_path.c_str());

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;

            fprintf(stderr, "Received: %s\n", line.c_str());

            try {
                json request = json::parse(line);

                if (!request.contains("jsonrpc") || request["jsonrpc"] != "2.0") {
                    continue;
                }

                // All request must include a string or integer ID and this must not be null.
                json id = nullptr;
                if (request.contains("id")) {
                    id = request["id"];
                }
                std::string method = request.value("method", "");

                if (method == "initialize") {
                    handle_initialize(id, request.value("params", json::object()));
                } else if (method == "tools/list") {
                    handle_list_tools(id);
                } else if (method == "tools/call") {
                    handle_tool_call(id, request.value("params", json::object()));
                } else if (method == "notifications/initialized") {
                    fprintf(stderr, "Client initialization completed\n");
                } else {
                    send_error(id, -32601, "Method not found: " + method);
                }

            } catch (const json::parse_error & e) {
                fprintf(stderr, "JSON parse error: %s\n", e.what());
            } catch (const std::exception & e) {
                fprintf(stderr, "Error processing request: %s\n", e.what());
            }
        }
    }
};

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    fprintf(stderr, "Whisper MCP Server starting...\n");

    whisper_params wparams;
    mcp_params     mparams;

    if (whisper_params_parse(argc, argv, wparams, mparams) == false) {
        whisper_print_usage(argc, argv, wparams, mparams);
        return 1;
    }

    if (wparams.language != "auto" && whisper_lang_id(wparams.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", wparams.language.c_str());
        whisper_print_usage(argc, argv, wparams, mparams);
        exit(0);
    }

    if (mparams.ffmpeg_converter) {
        check_ffmpeg_availibility();
    }

    WhisperMCPServer server(mparams, wparams, wparams.model);
    server.run();

    return 0;
}
