#pragma once

#include "mcp_transport.hpp"
#include "json.hpp"
#include "whisper.h"
#include <string>

using json = nlohmann::ordered_json;

// Forward declarations to avoid including heavy headers
struct mcp_params;
struct whisper_params;

/**
 * Whisper MCP Server Handler
 * 
 * Implements the Model Context Protocol for Whisper speech-to-text functionality.
 * Provides tools for:
 * - Audio transcription via whisper.cpp
 * - Model information queries
 * 
 * This handler processes MCP JSON-RPC messages and delegates responses
 * to the configured transport layer.
 */
class WhisperMCPHandler {
public:
    explicit WhisperMCPHandler(MCPTransport* transport, 
                              const struct mcp_params& mparams,
                              const struct whisper_params& wparams, 
                              const std::string& model_path);
    ~WhisperMCPHandler();

    /**
     * Process an incoming MCP message.
     * @param request JSON-RPC 2.0 request object
     * @return true if message was handled, false if invalid format
     */
    bool handle_message(const json& request);

private:
    // MCP protocol methods
    void handle_initialize(const json& id, const json& params);
    void handle_list_tools(const json& id);
    void handle_tool_call(const json& id, const json& params);
    void handle_notification_initialized();

    // Response helpers
    void send_result(const json& id, const json& result);
    void send_error(const json& id, int code, const std::string& message);

    // Whisper-specific functionality
    bool load_model();
    std::string transcribe_file(const std::string& filepath, 
                               const std::string& language = "auto", 
                               bool translate = false);
    bool load_audio_file(const std::string& fname_inp, std::vector<float>& pcmf32);
    
    // Tool implementations
    json create_transcribe_tool_result(const json& arguments);
    json create_model_info_tool_result();

    // State
    MCPTransport* transport_;
    struct whisper_context* ctx_;
    std::string model_path_;
    bool model_loaded_;
    struct mcp_params mparams_;
    struct whisper_params wparams_;
};
