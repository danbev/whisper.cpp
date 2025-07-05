#pragma once

#include "mcp-transport.hpp"
#include "mcp-params.hpp"

#include "whisper.h"
#include <string>
#include <vector>

class WhisperMCPHandler {
public:
    explicit WhisperMCPHandler(MCPTransport * transport, 
                              const struct mcp_params & mparams,
                              const struct whisper_params & wparams, 
                              const std::string & model_path);
    ~WhisperMCPHandler();

    // Process incoming MCP message
    bool handle_message(const json & request);

private:
    // MCP protocol methods
    void handle_initialize(const json & id, const json & params);
    void handle_list_tools(const json & id);
    void handle_tool_call(const json & id, const json & params);
    void handle_notification_initialized();

    // Response helpers
    void send_result(const json & id, const json & result);
    void send_error(const json & id, int code, const std::string & message);

    bool load_model();
    std::string transcribe_file(const std::string & filepath, 
                               const std::string & language = "auto", 
                               bool translate = false);
    bool load_audio_file(const std::string & fname_inp, std::vector<float> & pcmf32);
    
    json create_transcribe_result(const json & arguments);
    json create_model_info_result();

    MCPTransport * transport_;
    struct whisper_context * ctx_;
    std::string model_path_;
    bool model_loaded_;
    struct mcp_params mparams_;
    struct whisper_params wparams_;
};
