#pragma once

#include "mcp-transport.hpp"

class WhisperMCPHandler;

class StdioTransport : public MCPTransport {
public:
    StdioTransport() = default;
    ~StdioTransport() = default;

    void send_response(const json & response) override;
    
    void run(WhisperMCPHandler * handler);
};
